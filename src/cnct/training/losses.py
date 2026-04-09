"""Hybrid reconstruction loss for the dual-domain cascade.

Verbatim port of the legacy ``3dunet/dual_domain_train.py`` loss stack:
L1 + 3D SSIM + 3D Sobel edge loss. Every kernel value, window size, and
weighting coefficient matches the original — the math-preservation
requirement for cnct.models extends to the training objective because a
perturbed loss would silently change the optimisation trajectory.

Total loss::

    L = alpha * L1 + beta * (1 - SSIM_3D) + gamma * L_edge_3D

The three terms are complementary:
    * L1 — pixel fidelity, robust to streak-artifact outliers.
    * SSIM — perceptual structure, enforces local contrast agreement.
    * Sobel edge — gradient-magnitude matching, suppresses spurious edges
      from missing-view artifacts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schema import LossCfg

__all__ = ["SSIM3DLoss", "Edge3DLoss", "HybridLoss", "build_loss"]


class SSIM3DLoss(nn.Module):
    """Structural similarity loss for 3D single-channel volumes.

    Uses uniform box averaging (``F.avg_pool3d``) for the local means
    and variances. Assumes both inputs lie in ``[-1, 1]`` so the data
    range constant ``L = 2``.

    Args:
        window_size: Side length of the cubic averaging window. Must be
            odd so the window has a well-defined centre voxel.
    """

    def __init__(self, window_size: int = 7) -> None:
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd; got {window_size}"
            )
        self.window_size = window_size
        self._k1_sq = 0.01 ** 2
        self._k2_sq = 0.03 ** 2

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return ``1 - mean(SSIM_map)`` for a batched 5-D input pair.

        Args:
            pred: Shape ``[B, 1, D, H, W]``, range ``[-1, 1]``.
            target: Same shape and range as ``pred``.

        Returns:
            Scalar SSIM loss tensor on the same device as the inputs.
        """
        data_range = 2.0  # [-1, 1] → L = 2
        C1 = self._k1_sq * data_range ** 2
        C2 = self._k2_sq * data_range ** 2

        ws = self.window_size
        pad = ws // 2

        mu_p = F.avg_pool3d(pred, ws, stride=1, padding=pad)
        mu_t = F.avg_pool3d(target, ws, stride=1, padding=pad)

        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_pp = F.avg_pool3d(pred * pred, ws, stride=1, padding=pad) - mu_pp
        sigma_tt = F.avg_pool3d(target * target, ws, stride=1, padding=pad) - mu_tt
        sigma_pt = F.avg_pool3d(pred * target, ws, stride=1, padding=pad) - mu_pt

        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
            (mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2)
        )
        return 1.0 - ssim_map.mean()


class Edge3DLoss(nn.Module):
    """3D Sobel-based edge-map L1 loss.

    Applies three non-learnable 3×3×3 Sobel kernels (one per spatial axis)
    to both prediction and target and returns the L1 distance between the
    resulting 3-channel edge maps. The kernels are stored as buffers so
    they move with the module onto any device.
    """

    def __init__(self) -> None:
        super().__init__()
        sobel_z = torch.tensor(
            [
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            ],
            dtype=torch.float32,
        )
        sobel_y = torch.tensor(
            [
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            ],
            dtype=torch.float32,
        )
        sobel_x = torch.tensor(
            [
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            ],
            dtype=torch.float32,
        )
        kernel = torch.stack([sobel_z, sobel_y, sobel_x], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """L1 distance between 3D Sobel edge maps.

        Args:
            pred: Shape ``[B, 1, D, H, W]``.
            target: Same shape as ``pred``.

        Returns:
            Scalar loss tensor.
        """
        edges_pred = F.conv3d(pred, self.kernel, padding=1)
        edges_target = F.conv3d(target, self.kernel, padding=1)
        return F.l1_loss(edges_pred, edges_target)


class HybridLoss(nn.Module):
    """Weighted sum of L1, 3D SSIM, and 3D edge losses.

    Args:
        alpha: L1 weight.
        beta: SSIM weight.
        gamma: Edge weight.
        ssim_window: SSIM window side length (must be odd).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 1.0,
        ssim_window: int = 7,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()
        self.ssim = SSIM3DLoss(window_size=ssim_window)
        self.edge = Edge3DLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the weighted total loss.

        Args:
            pred: Predicted volume, ``[B, 1, D, H, W]``.
            target: Ground-truth volume, ``[B, 1, D, H, W]``.

        Returns:
            Scalar total loss tensor.
        """
        l1 = self.l1(pred, target)
        ssim = self.ssim(pred, target)
        edge = self.edge(pred, target)
        return self.alpha * l1 + self.beta * ssim + self.gamma * edge


def build_loss(cfg: LossCfg) -> HybridLoss:
    """Construct a :class:`HybridLoss` from a validated :class:`LossCfg`.

    Args:
        cfg: Validated loss configuration.

    Returns:
        Instantiated loss module (not yet moved to any device).
    """
    return HybridLoss(
        alpha=cfg.alpha,
        beta=cfg.beta,
        gamma=cfg.gamma,
        ssim_window=cfg.ssim_window,
    )
