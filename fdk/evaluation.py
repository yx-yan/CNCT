#!/usr/bin/env python3
"""Evaluate FDK reconstructions against ground-truth CT volumes.

Pipeline stage 3/3.  For each case with recon_fdk.npy in FDK_DIR:
  1. Load the ground-truth NIfTI and convert HU to mu (same units as FDK)
  2. Compute volumetric PSNR and mean per-slice SSIM
  3. Optionally save side-by-side comparison images (GT vs FDK vs difference)
  4. Write a CSV summary of all evaluated cases

Input:  FDK_DIR/<case>/recon_fdk.npy  (from fdk.py)
Output: EVAL_DIR/evaluation_results.csv
"""

import os
import glob
import csv

import numpy as np

from config import DATA_DIR, FDK_DIR, EVAL_DIR, MAX_CASES, CASE_START, CASE_END, IMAGE_DPI, SAVE_PNG
from eval_utils import load_gt_as_mu, compute_psnr_ssim, save_comparison


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

cases = sorted(glob.glob(os.path.join(DATA_DIR, "*.nii.gz")))[:MAX_CASES][CASE_START:CASE_END]
print(f"Found {len(cases)} cases\n")

results = []

for nii_path in cases:
    case_name = os.path.basename(nii_path).replace("_0000.nii.gz", "")
    recon_path = os.path.join(FDK_DIR, case_name, "recon_fdk.npy")
    case_out = os.path.join(EVAL_DIR, case_name)

    if not os.path.exists(recon_path):
        print(f"  Skipping {case_name} — recon_fdk.npy not found")
        continue

    print(f"=== Evaluating {case_name} ===")

    gt, dVoxel = load_gt_as_mu(nii_path)
    recon = np.load(recon_path).astype(np.float32)
    print(f"  GT shape: {gt.shape}, Recon shape: {recon.shape}")

    if gt.shape != recon.shape:
        print(f"  WARNING: shape mismatch — skipping metrics")
        continue

    psnr, ssim = compute_psnr_ssim(gt, recon)
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
    results.append({"case": case_name, "psnr_db": round(psnr, 4), "ssim": round(ssim, 6)})

    if SAVE_PNG:
        os.makedirs(case_out, exist_ok=True)
        save_comparison(gt, recon, dVoxel, case_out, case_name, "FDK Reconstruction", IMAGE_DPI,
                        psnr=psnr, ssim=ssim)
        print(f"  Comparison images saved to {case_out}/")

# --- CSV summary ---
if results:
    os.makedirs(EVAL_DIR, exist_ok=True)
    csv_path = os.path.join(EVAL_DIR, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "psnr_db", "ssim"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    psnr_vals = [r["psnr_db"] for r in results]
    ssim_vals = [r["ssim"] for r in results]
    print(f"\nSummary across {len(results)} cases:")
    print(f"  PSNR — mean: {np.mean(psnr_vals):.2f} dB, std: {np.std(psnr_vals):.2f} dB")
    print(f"  SSIM — mean: {np.mean(ssim_vals):.4f}, std: {np.std(ssim_vals):.4f}")

print("\nEvaluation done.")