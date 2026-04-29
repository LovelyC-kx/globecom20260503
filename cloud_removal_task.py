"""
Per-satellite cloud-removal training task (v1).

Analogue of `EuroSatSNNTask` from learning_task.py but for:
  • regression (cloudy → clear)
  • VLIFNet backbone (SpikingJelly multi-step)
  • Charbonnier + (1-SSIM) joint loss
  • mandatory functional.reset_net() around every forward/backward pair
  • local optimizer state persisted ACROSS inter-plane rounds
    (FedAvg/AdaptiveFedOpt convention: client moment buffers are local)

Design notes
------------
1. The task holds its own `VLIFNet` and its own `AdamW` optimizer instance.
   `apply_global_weights(state_dict)` loads non-BN (or all, per aggregation
   policy) weights without resetting the optimizer moment buffers.

2. Forward pass is wrapped by `_with_reset(...)` so that SpikingJelly's
   hidden state is cleared before AND after each forward.  This is the
   pattern used in upstream VLIFNet train.py and the SpikingJelly
   sequential-RSNN example.

3. State-dict copy-out (`get_weights()`) is a deep copy on CPU to avoid
   carrying GPU graph references into the aggregation path.

4. The module intentionally does NOT call `deepcopy(self.model)`: that
   pattern in the original FLSNN code silently carries SNN buffers
   (membrane potential, running stats) into the aggregation step.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import functional

from Spiking_Models.VLIFNet import build_vlifnet


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Differentiable L1-approx; ε=1e-3 per Lai et al. 2018.

    Charbonnier(x, y) = mean( sqrt((x - y)^2 + ε^2) )

    Twice differentiable — preserves the L-smoothness assumption used in
    Thm 2 of the FLSNN paper.
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


def _gaussian_window(window_size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


class SSIMLoss(nn.Module):
    """Standard SSIM on [0, 1] RGB images; loss = 1 - SSIM.

    Implemented from scratch to avoid the upstream utils.SSIM which holds
    a non-serialisable window buffer on __init__ (breaks DataParallel).
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = C1
        self.C2 = C2
        # Window is materialised lazily per-device to be DataParallel-safe.

    def _build_window(self, channel: int, device, dtype) -> torch.Tensor:
        g1d = _gaussian_window(self.window_size, self.sigma, device).to(dtype)
        w2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)     # [ws, ws]
        return w2d.expand(channel, 1, self.window_size, self.window_size).contiguous()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"SSIMLoss expects [B, C, H, W]; got {x.shape}"
        B, C, H, W = x.shape
        window = self._build_window(C, x.device, x.dtype)
        pad = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=pad, groups=C)
        mu_y = F.conv2d(y, window, padding=pad, groups=C)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

        ssim_num = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        ssim_den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        ssim_map = ssim_num / ssim_den
        return 1.0 - ssim_map.mean()


class CloudLoss(nn.Module):
    """v1 composite loss: Charbonnier + λ·(1-SSIM)."""
    def __init__(self, ssim_weight: float = 0.1, eps: float = 1e-3):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=eps)
        self.ssim = SSIMLoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ch = self.charbonnier(pred, target)
        ss = self.ssim(pred, target)
        return ch + self.ssim_weight * ss


# ---------------------------------------------------------------------------
# SpikingJelly helpers
# ---------------------------------------------------------------------------

# Re-exported from aggregation.py to keep a single source of truth.
from aggregation import is_bn_key, apply_aggregated  # noqa: E402,F401


def _reset_snn(model: nn.Module) -> None:
    """Clear all SpikingJelly hidden states (membrane potentials, etc.)."""
    functional.reset_net(model)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class CloudRemovalSNNTask:
    """One satellite's trainer.

    Parameters
    ----------
    args : argparse.Namespace-like
        Required fields:
          .lr, .wd, .train_batch_size, .num_workers,
          .local_iters, .T, .vlif_dim, .en_blocks, .de_blocks,
          .ssim_weight, .charbonnier_eps,
          .use_amp (bool; v1 False).
    local_dataset : torch.utils.data.Dataset
        The satellite's local training set (Subset of PairedCloudDataset).
    init_state_dict : dict | None
        Optional initial weights (all satellites receive a common init
        before round 0).
    device : str | torch.device
    """

    def __init__(self,
                 args,
                 local_dataset: Dataset,
                 init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
                 device: str = "cuda"):
        self.args = args
        self.device = torch.device(device)

        # -- Model ---------------------------------------------------------
        backend = getattr(args, "vlif_backend", "torch")
        self.model = build_vlifnet(
            dim=args.vlif_dim,
            en_num_blocks=tuple(args.en_blocks),
            de_num_blocks=tuple(args.de_blocks),
            T=args.T,
            use_refinement=False,
            inp_channels=3, out_channels=3,
            backend=backend,
        ).to(self.device)

        if init_state_dict is not None:
            # strict=True by design — we want to catch architecture drift early.
            self.model.load_state_dict(init_state_dict, strict=True)

        # -- Data ----------------------------------------------------------
        self.local_dataset = local_dataset
        self.loader = DataLoader(
            local_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=getattr(args, "num_workers", 2),
            persistent_workers=False,
        )

        # -- Optimizer (persisted across inter-plane rounds) ---------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=getattr(args, "wd", 0.0),
        )
        self.clip_grad = getattr(args, "clip_grad", 1.0)

        # -- Loss ----------------------------------------------------------
        self.criterion = CloudLoss(
            ssim_weight=args.ssim_weight,
            eps=args.charbonnier_eps,
        ).to(self.device)

        # -- Running stats -------------------------------------------------
        self.last_train_loss: float = math.nan
        self.last_train_charbonnier: float = math.nan
        self.last_train_ssim: float = math.nan
        self.global_round: int = 0

    # ------------------------------------------------------------------
    # Weight I/O (used by aggregation)
    # ------------------------------------------------------------------

    def get_weights(self, cpu: bool = True) -> Dict[str, torch.Tensor]:
        """Return a detached copy of the model state-dict.

        SpikingJelly's MemoryModule writes non-tensor values (scalar
        floats for un-initialised neuron membrane potentials, `None`
        sentinels) into the state_dict alongside real tensors.  We must
        copy tensors and pass the sentinels through verbatim; calling
        `.detach()` / `.clone()` / `.cpu()` on a float/None raises.
        """
        sd = self.model.state_dict()
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                if cpu:
                    out[k] = v.detach().cpu().clone()
                else:
                    out[k] = v.detach().clone()
            else:
                # Scalar float / None sentinel from SpikingJelly — preserve as-is.
                out[k] = v
        return out

    def apply_global_weights(self,
                             global_state: Dict[str, torch.Tensor],
                             bn_local: bool = False) -> None:
        """Copy aggregated weights into the model in-place.

        If `bn_local=True`, BN-tagged keys are **skipped** — the
        satellite keeps its own BN running stats + affine params
        (FedBN, ICLR 2021).  v1 always passes bn_local=False.
        """
        apply_aggregated(self.model.state_dict(), global_state, bn_local=bn_local)
        # Critical: the weights we just loaded may be the average of
        # several noisy clients — clear SNN hidden state before next fwd.
        _reset_snn(self.model)

    # ------------------------------------------------------------------
    # Training / Inference
    # ------------------------------------------------------------------

    def _cosine_lr(self, step: int, total_steps: int, warmup_steps: int) -> float:
        base_lr = self.args.lr
        min_lr  = getattr(self.args, "min_lr", 1e-7)
        if step < warmup_steps:
            return base_lr * (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = max(0.0, min(1.0, t))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    def local_training(self,
                       total_global_rounds: int,
                       warmup_rounds: int = 3) -> Tuple[float, float, float]:
        """Run `args.local_iters` passes over the local dataset.

        Returns (avg_total_loss, avg_charbonnier, avg_ssim_loss).
        """
        self.model.train()
        n_batches = 0
        sum_loss = sum_ch = sum_ss = 0.0

        # Cosine LR with warmup, shared clock across all satellites
        lr = self._cosine_lr(self.global_round, total_global_rounds, warmup_rounds)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        for _ in range(self.args.local_iters):
            for cloudy, clear in self.loader:
                cloudy = cloudy.to(self.device, non_blocking=True)
                clear  = clear.to(self.device,  non_blocking=True)

                _reset_snn(self.model)
                self.optimizer.zero_grad(set_to_none=True)

                pred = self.model(cloudy)                 # [B, 3, H, W]
                loss_ch = self.criterion.charbonnier(pred, clear)
                loss_ss = self.criterion.ssim(pred, clear)
                loss = loss_ch + self.criterion.ssim_weight * loss_ss

                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                _reset_snn(self.model)                     # per R1

                n_batches += 1
                sum_loss += loss.item()
                sum_ch   += loss_ch.item()
                sum_ss   += loss_ss.item()

        if n_batches == 0:
            # Should not happen in v1 (IID partition guarantees ≥1 sample)
            return math.nan, math.nan, math.nan

        self.last_train_loss        = sum_loss / n_batches
        self.last_train_charbonnier = sum_ch   / n_batches
        self.last_train_ssim        = sum_ss   / n_batches
        return self.last_train_loss, self.last_train_charbonnier, self.last_train_ssim

    @torch.no_grad()
    def forward_on(self, cloudy: torch.Tensor) -> torch.Tensor:
        """Pure inference helper; caller manages reset_net bookkeeping."""
        self.model.eval()
        _reset_snn(self.model)
        out = self.model(cloudy.to(self.device, non_blocking=True))
        _reset_snn(self.model)
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup_between_rounds(self) -> None:
        """Release transient tensors; keep optimizer state + weights."""
        torch.cuda.empty_cache()
        self.global_round += 1
