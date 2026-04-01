#!/usr/bin/env python3
"""
Convert FDK reconstructions (.npy) and ground-truth CT volumes (.nii.gz)
into HDF5 files for pytorch-3dunet training.

  Raw   = recon_fdk.npy  — FDK reconstruction (linear attenuation μ, Z×Y×X)
  Label = *_0000.nii.gz  — ground-truth CT (HU, X×Y×Z → converted to μ,
                           transposed to Z×Y×X)

Output: OUT_DIR/{train,val,test}/Case_XXXXX.h5
Each HDF5 file contains:
  raw   : float32 (Z, Y, X) — FDK in μ units
  label : float32 (Z, Y, X) — GT converted from HU to μ units

Usage:
  python prepare_data.py [--fdk_dir ...] [--gt_dir ...] [--out_dir ...] [--dry_run]
"""

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from geometry import hu_to_mu

# ── Defaults (override via CLI) ───────────────────────────────────────────────
FDK_DIR    = "/projects/CTdata/fdk60"
GT_DIR     = "/projects/CTdata/AbdomenCT-1K-ImagePart1"
OUT_DIR    = "/projects/CTdata/h5_3dunet"
MU_WATER   = 0.02    # mm-1  (must match config.py)
PATCH_MIN  = 64      # skip cases where any spatial dim < this
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC = 1 - TRAIN_FRAC - VAL_FRAC = 0.10


def main():
    parser = argparse.ArgumentParser(description="Prepare HDF5 dataset for 3D UNet.")
    parser.add_argument("--fdk_dir",   default=FDK_DIR,   help="FDK .npy root dir")
    parser.add_argument("--gt_dir",    default=GT_DIR,    help="Ground-truth NIfTI root dir")
    parser.add_argument("--out_dir",   default=OUT_DIR,   help="HDF5 output root dir")
    parser.add_argument("--mu_water",  type=float, default=MU_WATER)
    parser.add_argument("--patch_min", type=int,   default=PATCH_MIN,
                        help="Skip cases with any spatial dimension < this value")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print split without writing any files")
    args = parser.parse_args()

    fdk_dir = Path(args.fdk_dir)
    gt_dir  = Path(args.gt_dir)
    out_dir = Path(args.out_dir)

    # ── Discover matched pairs ─────────────────────────────────────────────────
    fdk_cases = sorted(
        p.name for p in fdk_dir.iterdir()
        if p.is_dir() and (p / "recon_fdk.npy").exists()
    )
    print(f"Found {len(fdk_cases)} FDK cases in {fdk_dir}")

    pairs, missing = [], []
    for case_id in fdk_cases:
        nii_path = gt_dir / f"{case_id}_0000.nii.gz"
        if nii_path.exists():
            pairs.append((case_id,
                          fdk_dir / case_id / "recon_fdk.npy",
                          nii_path))
        else:
            missing.append(case_id)

    if missing:
        print(f"WARNING: {len(missing)} FDK cases have no matching NIfTI "
              f"(first 5: {missing[:5]})")
    print(f"Matched {len(pairs)} case pairs\n")

    # ── Deterministic train / val / test split ─────────────────────────────────
    # Fixed seed makes the split reproducible across runs.  The indices are
    # shuffled once and sliced — no case can appear in multiple splits.
    rng   = np.random.default_rng(args.seed)
    idx   = rng.permutation(len(pairs))
    n_tr  = int(len(pairs) * TRAIN_FRAC)
    n_val = int(len(pairs) * VAL_FRAC)

    splits = {
        "train": [pairs[i] for i in idx[:n_tr]],
        "val":   [pairs[i] for i in idx[n_tr : n_tr + n_val]],
        "test":  [pairs[i] for i in idx[n_tr + n_val :]],
    }
    for split, sp in splits.items():
        print(f"  {split:5s}: {len(sp)} cases")

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    # ── Convert and write HDF5 ─────────────────────────────────────────────────
    skipped = 0
    for split, case_list in splits.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for case_id, fdk_path, nii_path in case_list:
            out_path = split_dir / f"{case_id}.h5"
            # Idempotent: skip cases already written so re-runs are safe.
            if out_path.exists():
                print(f"  [skip existing] {split}/{out_path.name}")
                continue

            # Raw: FDK reconstruction — fdk.py already saves as (Z, Y, X), μ units.
            fdk = np.load(fdk_path).astype(np.float32)

            # Label: NIfTI GT — nibabel's get_fdata() returns (X, Y, Z) following
            # the NIfTI/RAS convention.  Transpose to (Z, Y, X) to match TIGRE's
            # axis order used throughout this pipeline.
            nii    = nib.load(str(nii_path))
            gt_hu  = nii.get_fdata(dtype=np.float32)                    # (X, Y, Z)
            gt_mu  = hu_to_mu(np.transpose(gt_hu, (2, 1, 0)),           # (Z, Y, X)
                               mu_water=args.mu_water)

            # Both volumes must have the same shape: the FDK was reconstructed on
            # the same voxel grid as the original CT, so a mismatch indicates a
            # data preparation error (e.g. wrong geometry or NIfTI mismatch).
            if fdk.shape != gt_mu.shape:
                print(f"  [SHAPE MISMATCH] {case_id}: fdk={fdk.shape} "
                      f"gt={gt_mu.shape} — skipping")
                skipped += 1
                continue

            # patch_min (default 64) matches the Z dimension of patch_shape in
            # train_config.yaml.  If any spatial dimension is smaller than the
            # patch, SliceBuilder cannot extract even a single patch and the
            # case is useless for training.
            if min(fdk.shape) < args.patch_min:
                print(f"  [TOO SMALL] {case_id}: shape={fdk.shape} — skipping "
                      f"(min dim {min(fdk.shape)} < {args.patch_min})")
                skipped += 1
                continue

            # gzip level 4 is a reasonable speed/size tradeoff for float32
            # volumetric data.  CT volumes compress well (~3–5×) due to
            # large uniform regions (air, tissue).
            with h5py.File(out_path, "w") as f:
                f.create_dataset("raw",   data=fdk,   compression="gzip",
                                 compression_opts=4)
                f.create_dataset("label", data=gt_mu, compression="gzip",
                                 compression_opts=4)

            print(f"  [{split}] {case_id}  shape={fdk.shape}")

    print(f"\nDone. Skipped {skipped} cases.")
    for split in splits:
        n = len(list((out_dir / split).glob("*.h5")))
        print(f"  {split}: {n} HDF5 files in {out_dir / split}")


if __name__ == "__main__":
    main()
