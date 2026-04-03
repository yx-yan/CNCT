#!/usr/bin/env python3
"""Evaluate 3D UNet predictions against ground-truth CT volumes.

For each postprocessed prediction (<CaseID>_recon.npy) in PRED_DIR:
  1. Load the ground-truth NIfTI and convert HU to mu (same units as prediction)
  2. Compute volumetric PSNR and mean per-slice SSIM
  3. Optionally save side-by-side comparison images (GT vs UNet vs difference)
  4. Write a CSV summary of all evaluated cases

Input:  /projects/CTdata/3dunet_predictions/<CaseID>_recon.npy  (from postprocess_predictions.py)
Output: UNET_EVAL_DIR/evaluation_results.csv
        UNET_EVAL_DIR/<CaseID>/eval_{axial,coronal,sagittal}.png  (if SAVE_PNG)
"""

import argparse
import os
import glob
import csv

import numpy as np

from config import DATA_DIR, UNET_EVAL_DIR, IMAGE_DPI, SAVE_PNG
from eval_utils import load_gt_as_mu, compute_psnr_ssim, save_comparison


PRED_DIR = "/projects/CTdata/3dunet_predictions"

_parser = argparse.ArgumentParser()
_parser.add_argument("--pred_dir", default=PRED_DIR, help="Directory with *_recon.npy files")
_parser.add_argument("--eval_dir", default=UNET_EVAL_DIR, help="Output directory for eval results")
_args = _parser.parse_args()

PRED_DIR = _args.pred_dir
UNET_EVAL_DIR = _args.eval_dir


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*_recon.npy")))
print(f"Found {len(pred_files)} prediction file(s) in {PRED_DIR}\n")

results = []

for pred_path in pred_files:
    case_name = os.path.basename(pred_path).replace("_predictions_recon.npy", "").replace("_recon.npy", "")
    nii_path = os.path.join(DATA_DIR, f"{case_name}_0000.nii.gz")
    case_out = os.path.join(UNET_EVAL_DIR, case_name)

    if not os.path.exists(nii_path):
        print(f"  Skipping {case_name} — NIfTI not found at {nii_path}")
        continue

    print(f"=== Evaluating {case_name} ===")

    gt, dVoxel = load_gt_as_mu(nii_path)
    recon = np.load(pred_path).astype(np.float32)
    print(f"  GT shape: {gt.shape}, Prediction shape: {recon.shape}")

    if gt.shape != recon.shape:
        print(f"  WARNING: shape mismatch — skipping metrics")
        continue

    psnr, ssim = compute_psnr_ssim(gt, recon)
    print(f"  PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}")
    results.append({"case": case_name, "psnr_db": round(psnr, 4), "ssim": round(ssim, 6)})

    if SAVE_PNG:
        os.makedirs(case_out, exist_ok=True)
        save_comparison(gt, recon, dVoxel, case_out, case_name, "UNet Prediction", IMAGE_DPI)
        print(f"  Comparison images saved to {case_out}/")

# --- CSV summary ---
if results:
    os.makedirs(UNET_EVAL_DIR, exist_ok=True)
    csv_path = os.path.join(UNET_EVAL_DIR, "evaluation_results.csv")
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