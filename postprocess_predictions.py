#!/usr/bin/env python3
"""
Convert pytorch-3dunet prediction HDF5 files back to numpy (.npy) arrays.

pytorch-3dunet's StandardPredictor saves predictions as HDF5 files
(dataset "predictions") in the output_dir specified in test_config.yaml.
The predictions are in normalised [-1, 1] space; this script inverts the
per-volume Normalize transform and converts μ → HU for comparison.

Input : /projects/CTdata/3dunet_predictions/<CaseID>.h5
            dataset "predictions" — shape (1, Z, Y, X), float32, in [-1, 1]
Original raw stats are re-read from the corresponding test HDF5 file so the
inverse normalisation is exact.

Output: /projects/CTdata/3dunet_predictions/<CaseID>_recon.npy
            float32 (Z, Y, X), μ units  (same as recon_fdk.npy)
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from geometry import mu_to_hu

PRED_DIR  = "/projects/CTdata/3dunet_predictions"
H5_TEST   = "/projects/CTdata/h5_3dunet/test"
MU_WATER  = 0.02   # mm-1


def normalize_inverse(pred_norm: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    """Invert pytorch-3dunet's Normalize transform: [-1, 1] → original μ range.

    pytorch-3dunet's Normalize maps a volume v to [-1, 1] as:
        v_norm = 2 * (v - pmin) / (pmax - pmin) - 1

    where pmin = percentile(v, 1) and pmax = percentile(v, 99.6) of the raw
    (FDK input) volume.  Rearranging for v:
        v = (v_norm + 1) / 2 * (pmax - pmin) + pmin

    The percentiles are taken from the *raw* (FDK) dataset, not the label,
    because the network is trained to predict in the raw input's normalised
    space.  postprocess_predictions.py must re-read those same raw stats from
    the source HDF5 file to invert exactly.

    Using 1st/99.6th percentiles (rather than min/max) makes the scaling
    robust to isolated outlier voxels in the FDK reconstruction.
    """
    return (pred_norm + 1.0) / 2.0 * (pmax - pmin) + pmin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default=PRED_DIR)
    parser.add_argument("--h5_test",  default=H5_TEST)
    parser.add_argument("--mu_water", type=float, default=MU_WATER)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    h5_test  = Path(args.h5_test)

    pred_files = sorted(pred_dir.glob("*.h5"))
    if not pred_files:
        print(f"No prediction HDF5 files found in {pred_dir}")
        return

    print(f"Found {len(pred_files)} prediction file(s) in {pred_dir}")

    for pred_path in pred_files:
        case_id   = pred_path.stem           # e.g. "Case_00321"
        raw_h5    = h5_test / f"{case_id}.h5"
        out_path  = pred_dir / f"{case_id}_recon.npy"

        if out_path.exists():
            print(f"  [skip existing] {out_path.name}")
            continue

        with h5py.File(pred_path, "r") as f:
            if "predictions" not in f:
                print(f"  [MISSING 'predictions'] {pred_path.name} — skipping")
                continue
            # StandardPredictor stores predictions with an explicit channel dim:
            # shape (1, Z, Y, X).  squeeze() removes it → (Z, Y, X).
            pred_norm = f["predictions"][:]    # (1, Z, Y, X) or (Z, Y, X)

        pred_norm = pred_norm.squeeze()        # → (Z, Y, X)

        # Re-read the raw volume to recover the pmin/pmax used during Normalize.
        # (pytorch-3dunet's Normalize uses np.percentile(raw, 1) and (99.6))
        # These stats come from the *raw* dataset (FDK input), not the label,
        # because the network output is in the raw's normalised coordinate space.
        if raw_h5.exists():
            with h5py.File(raw_h5, "r") as f:
                raw = f["raw"][:]
            pmin = float(np.percentile(raw, 1))
            pmax = float(np.percentile(raw, 99.6))
            pred_mu = normalize_inverse(pred_norm, pmin, pmax)
        else:
            print(f"  [WARNING] raw HDF5 not found for {case_id}; "
                  f"saving normalised output as-is ([-1,1] range)")
            pred_mu = pred_norm

        np.save(out_path, pred_mu.astype(np.float32))
        print(f"  Saved {out_path.name}  shape={pred_mu.shape}  "
              f"range=[{pred_mu.min():.4f}, {pred_mu.max():.4f}]")

    print("Done.")


if __name__ == "__main__":
    main()
