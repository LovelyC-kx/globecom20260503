"""
Full-image PSNR / SSIM evaluation for v1 cloud-removal pipeline.

Two evaluation modes
--------------------
1. `evaluate_fullimage(model, loader)` — feeds each test image at its
   native resolution (default for CUHK-CR 512×512 is fine on V100).

2. `evaluate_sliding(model, loader, window, stride)` — memory-safe fallback
   when native-resolution inference does not fit in GPU memory.
   Predictions on overlapping windows are blended with a raised-cosine
   weight to hide seams.

Both modes:
    * run the model in eval mode;
    * call functional.reset_net before and after each forward, per R1;
    * clip predictions to [0, 1] before metric computation (PSNR
      definition assumes bounded range).

Metrics reported (per image, then averaged):
    PSNR (dB)       : 20·log10(1 / RMSE)
    SSIM            : standard Gaussian-window SSIM, window=11, σ=1.5

Returns an EvalResult namedtuple with per-image arrays + their means
for downstream plotting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from cloud_removal_task import SSIMLoss


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    psnr_per_image: List[float] = field(default_factory=list)
    ssim_per_image: List[float] = field(default_factory=list)

    @property
    def mean_psnr(self) -> float:
        return float(np.mean(self.psnr_per_image)) if self.psnr_per_image else float("nan")

    @property
    def mean_ssim(self) -> float:
        return float(np.mean(self.ssim_per_image)) if self.ssim_per_image else float("nan")

    def summary(self) -> str:
        n = len(self.psnr_per_image)
        return (f"n={n}  "
                f"PSNR mean={self.mean_psnr:.4f}  "
                f"SSIM mean={self.mean_ssim:.4f}")


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------

def _torch_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR in dB for images in [0, 1]; both inputs must share shape [C,H,W]."""
    pred   = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = (pred - target).pow(2).mean()
    if mse.item() == 0:
        return torch.tensor(99.0, device=pred.device)
    return 20.0 * torch.log10(1.0 / mse.sqrt())


# Re-use the same SSIM implementation as training loss (1 - SSIM).
# We need SSIM itself rather than the loss → compute 1 - loss.
_SSIM_LOSS = SSIMLoss()


def _torch_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred   = pred.clamp(0.0, 1.0).unsqueeze(0)    # [1, C, H, W]
    target = target.clamp(0.0, 1.0).unsqueeze(0)
    with torch.no_grad():
        ssim_loss = _SSIM_LOSS(pred, target)
    return 1.0 - ssim_loss


# ---------------------------------------------------------------------------
# Sliding-window inference helper (R-v1-7 fallback)
# ---------------------------------------------------------------------------

def _raised_cosine_window(H: int, W: int, device, dtype) -> torch.Tensor:
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    wy = 0.5 - 0.5 * torch.cos(2 * math.pi * (y + 0.5) / H)
    wx = 0.5 - 0.5 * torch.cos(2 * math.pi * (x + 0.5) / W)
    return wy.unsqueeze(1) * wx.unsqueeze(0)  # [H, W]


@torch.no_grad()
def _sliding_forward(model: nn.Module, img: torch.Tensor,
                     window: int, stride: int,
                     device: torch.device) -> torch.Tensor:
    """img: [C, H, W] on device.  Returns restored [C, H, W]."""
    C, H, W = img.shape
    out  = torch.zeros_like(img)
    norm = torch.zeros((1, H, W), device=device, dtype=img.dtype)

    for y0 in list(range(0, max(1, H - window + 1), stride)) + [H - window]:
        for x0 in list(range(0, max(1, W - window + 1), stride)) + [W - window]:
            y0 = max(0, min(H - window, y0))
            x0 = max(0, min(W - window, x0))
            patch = img[:, y0:y0 + window, x0:x0 + window].unsqueeze(0)
            functional.reset_net(model)
            pred = model(patch)[0]
            functional.reset_net(model)
            w = _raised_cosine_window(window, window, device, img.dtype)
            out [:, y0:y0 + window, x0:x0 + window] += pred * w
            norm[:, y0:y0 + window, x0:x0 + window] += w

    return out / norm.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Public evaluation entry points
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fullimage(model: nn.Module,
                       loader: DataLoader,
                       device: Optional[torch.device] = None,
                       ) -> EvalResult:
    """Run the model at native resolution for every (cloudy, clear) pair."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    res = EvalResult()

    for batch in loader:
        cloudy, clear = batch
        cloudy = cloudy.to(device, non_blocking=True)
        clear  = clear.to(device,  non_blocking=True)

        functional.reset_net(model)
        pred = model(cloudy)
        functional.reset_net(model)

        for i in range(pred.shape[0]):
            res.psnr_per_image.append(_torch_psnr(pred[i], clear[i]).item())
            res.ssim_per_image.append(_torch_ssim(pred[i], clear[i]).item())

    return res


@torch.no_grad()
def evaluate_sliding(model: nn.Module,
                     loader: DataLoader,
                     window: int = 256,
                     stride: int = 128,
                     device: Optional[torch.device] = None,
                     ) -> EvalResult:
    """Run sliding-window inference; use when full-res OOM.

    `loader` must yield batch-size=1 to match the per-image sliding loop.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    res = EvalResult()

    for batch in loader:
        cloudy, clear = batch
        assert cloudy.shape[0] == 1, \
            "evaluate_sliding requires batch_size=1; got " + str(cloudy.shape[0])
        cloudy = cloudy[0].to(device, non_blocking=True)
        clear  = clear[0].to(device,  non_blocking=True)

        pred = _sliding_forward(model, cloudy, window, stride, device)

        res.psnr_per_image.append(_torch_psnr(pred, clear).item())
        res.ssim_per_image.append(_torch_ssim(pred, clear).item())

    return res


# ---------------------------------------------------------------------------
# Convenience: evaluate all plane models, average results
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_per_plane(constellation, loader: DataLoader,
                       *, window: int = 0, stride: int = 0,
                       device: Optional[torch.device] = None,
                       ) -> List[EvalResult]:
    """Evaluate each plane's current model on the same test loader.

    `window > 0` switches to sliding-window inference (memory-safe).
    """
    results: List[EvalResult] = []
    for p in range(constellation.num_planes):
        # All satellites in a plane share identical weights after aggregation
        model = constellation.planes[p][0].model
        if window and stride:
            # Need a batch_size=1 view of the loader.
            one_loader = DataLoader(
                loader.dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True)
            results.append(evaluate_sliding(model, one_loader, window, stride, device))
        else:
            results.append(evaluate_fullimage(model, loader, device))
    return results


def average_eval_results(results: List[EvalResult]) -> EvalResult:
    """Element-wise average across planes (same test set)."""
    if not results:
        return EvalResult()
    n_images = len(results[0].psnr_per_image)
    out = EvalResult()
    for i in range(n_images):
        out.psnr_per_image.append(float(np.mean([r.psnr_per_image[i] for r in results])))
        out.ssim_per_image.append(float(np.mean([r.ssim_per_image[i] for r in results])))
    return out
