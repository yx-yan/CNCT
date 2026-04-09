"""End-to-end orchestration for the FDK evaluation pipeline.

For each case, the runner:

    1. Loads the ground-truth NIfTI and converts HU to mu.
    2. Loads the FDK reconstruction (``recon_fdk.npy``).
    3. Computes volumetric PSNR and mean per-slice SSIM.
    4. Optionally saves axial / coronal / sagittal comparison PNGs.
    5. Appends the result to an in-memory list.

After the loop completes, results are written as CSV to
``cfg.eval_dir/evaluation_results.csv`` and aggregate statistics are logged.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config.schema import EvaluationCfg
from ..utils.io import ensure_dir, safe_load_npy
from ..utils.paths import case_nifti_path, case_output_dir, discover_cases
from .metrics import compute_psnr_ssim, load_gt_as_mu
from .visualize import save_comparison

logger = logging.getLogger(__name__)

_RECON_FILENAME = "recon_fdk.npy"
_RECON_LABEL = "FDK Reconstruction"
_CSV_FIELDS = ("case", "psnr_db", "ssim")


def evaluate_case(
    case_id: str,
    cfg: EvaluationCfg,
) -> Optional[Dict[str, float]]:
    """Evaluate a single case and optionally save comparison images.

    Args:
        case_id: Case identifier.
        cfg: Evaluation configuration.

    Returns:
        A dict with keys ``"case"``, ``"psnr_db"``, ``"ssim"`` on success;
        ``None`` if the case was skipped (missing file, shape mismatch, or
        load error).
    """
    nii_path = case_nifti_path(cfg.data_dir, case_id)
    recon_path = Path(cfg.fdk_dir) / case_id / _RECON_FILENAME

    if not recon_path.exists():
        logger.warning(
            "Skipping %s — %s not found", case_id, recon_path
        )
        return None

    logger.info("=== Evaluating %s ===", case_id)

    try:
        gt, dVoxel = load_gt_as_mu(nii_path, cfg.geometry.mu_water)
    except (FileNotFoundError, OSError) as exc:
        logger.error("Failed to load GT for %s: %s", case_id, exc)
        return None

    try:
        recon = safe_load_npy(recon_path).astype(np.float32)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error("Failed to load recon for %s: %s", case_id, exc)
        return None

    logger.info(
        "  GT shape=%s, recon shape=%s", gt.shape, recon.shape
    )

    try:
        psnr, ssim = compute_psnr_ssim(gt, recon)
    except ValueError as exc:
        logger.warning(
            "Shape mismatch for %s: %s — skipping metrics", case_id, exc
        )
        return None

    logger.info("  PSNR: %.2f dB  |  SSIM: %.4f", psnr, ssim)

    if cfg.save_png:
        try:
            case_out = case_output_dir(cfg.eval_dir, case_id)
            save_comparison(
                gt,
                recon,
                dVoxel,
                case_out,
                case_id,
                _RECON_LABEL,
                cfg.image_dpi,
                psnr=psnr,
                ssim=ssim,
            )
            logger.info("  Comparison images saved to %s", case_out)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to save comparison images for %s (continuing): %s",
                case_id,
                exc,
            )

    return {
        "case": case_id,
        "psnr_db": round(psnr, 4),
        "ssim": round(ssim, 6),
    }


def _write_csv_summary(
    results: List[Dict[str, float]],
    csv_path: Path,
) -> None:
    """Write the per-case evaluation results as a CSV file.

    Args:
        results: List of per-case result dicts.
        csv_path: Destination CSV path; parent directories are created.

    Raises:
        OSError: If the file cannot be written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        writer.writerows(results)


def run_evaluation(cfg: EvaluationCfg) -> None:
    """Orchestrate the full FDK evaluation pipeline across every case.

    Args:
        cfg: Evaluation configuration.
    """
    ensure_dir(cfg.eval_dir)
    cases = discover_cases(
        cfg.data_dir,
        start=cfg.case_start,
        end=cfg.case_end,
        max_cases=cfg.max_cases,
    )
    if not cases:
        logger.warning(
            "No cases found under %s; nothing to do", cfg.data_dir
        )
        return

    logger.info("Evaluating %d case(s)", len(cases))

    results: List[Dict[str, float]] = []
    for case_id in cases:
        result = evaluate_case(case_id, cfg)
        if result is not None:
            results.append(result)

    if not results:
        logger.warning(
            "No cases produced results; skipping CSV and summary"
        )
        return

    csv_path = Path(cfg.eval_dir) / "evaluation_results.csv"
    try:
        _write_csv_summary(results, csv_path)
        logger.info("Results saved to %s", csv_path)
    except OSError as exc:
        logger.error(
            "Failed to write CSV summary to %s: %s", csv_path, exc
        )

    psnr_vals = np.array([r["psnr_db"] for r in results], dtype=np.float64)
    ssim_vals = np.array([r["ssim"] for r in results], dtype=np.float64)
    logger.info(
        "Summary across %d case(s): "
        "PSNR mean=%.2f dB std=%.2f | SSIM mean=%.4f std=%.4f",
        len(results),
        float(psnr_vals.mean()),
        float(psnr_vals.std()),
        float(ssim_vals.mean()),
        float(ssim_vals.std()),
    )
