"""
v1 cloud-removal configuration.

Design choice
-------------
The legacy `config.py` runs argparse at module import time (args = parser.parse_args()),
which pollutes the test process every time anyone imports config.  For the v1
pipeline we avoid that footgun by exposing a single factory, `build_v1_args(...)`,
that returns an `argparse.Namespace` with all defaults pre-filled.  The smoke
script can optionally override fields with CLI flags via `parse_v1_cli()`.

Everything tuned for the agreed v1 config (see README §2.2 of the plan):

    * CUHK-CR1, RGB-only, patch=64 train / 512-full eval
    * 50/5/1 Walker Star constellation (num_planes=5, sats_per_plane=10)
    * VLIFNet dim=24, en=[2,2,4,4], de=[2,2,2,2], T=4, torch backend
    * AdamW 1e-3 + 3-epoch warmup + cosine LR
    * intra_plane_iters=2, local_iters=2, 10 global epochs
    * bn_local=False (FedAvg baseline — FedBN is v2)
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

V1_DEFAULTS: Dict[str, Any] = {
    # ---- Run identity ----
    "run_name":           "v1_smoke",
    "seed":               1234,
    "device":             "cuda:0",
    "output_dir":         "./Outputs",
    "tensorboard_dir":    "./Outputs/tb",

    # ---- Dataset ----
    "data_root":          "./data/CUHK-CR1",
    "dataset_name":       "CUHK-CR1",           # for logging only
    "use_rice_fallback":  False,                # if True, look in ./data/RICE1
    "train_split":        "train",              # sub-folder, or None = flat
    "test_split":         "test",
    "patch_size":         64,
    "train_batch_size":   4,                    # V100 16/32GB, dim=24
    "test_batch_size":    1,                    # full-res eval
    "num_workers":        2,

    # ---- Constellation ----
    "num_planes":         5,
    "sats_per_plane":     10,
    "partition_mode":     "iid",                # v1; 'dirichlet_cluster' is v2
    "partition_seed":     0,

    # ---- Model ----
    "vlif_dim":           24,
    "en_blocks":          [2, 2, 4, 4],
    "de_blocks":          [2, 2, 2, 2],
    "T":                  4,
    "vlif_backend":       "torch",              # 'cupy' optional
    "use_refinement":     False,

    # ---- Optimization ----
    "lr":                 1e-3,
    "min_lr":             1e-7,
    "wd":                 0.0,
    "warmup_epochs":      3,
    "clip_grad":          1.0,
    "use_amp":            False,                # v1 disabled (stability over speed)

    # ---- Training schedule ----
    "num_epoch":          10,                   # global epoch count (smoke)
    "intra_plane_iters":  2,
    "local_iters":        2,

    # ---- Loss ----
    "ssim_weight":        0.1,
    "charbonnier_eps":    1e-3,

    # ---- Aggregation ----
    "bn_local":           False,                # v2 flips to True for FedBN

    # ---- Evaluation ----
    "eval_every":         1,                    # global epochs between evals
    "eval_mode":          "fullimage",          # or "sliding"
    "sliding_window":     256,
    "sliding_stride":     128,
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_v1_args(**overrides: Any) -> argparse.Namespace:
    """Return a Namespace pre-filled with v1 defaults + any caller overrides."""
    cfg = dict(V1_DEFAULTS)
    cfg.update(overrides)
    ns = argparse.Namespace(**cfg)
    _validate(ns)
    return ns


def _validate(ns: argparse.Namespace) -> None:
    assert ns.num_planes > 0 and ns.sats_per_plane > 0
    assert len(ns.en_blocks) == 4 and len(ns.de_blocks) == 4
    assert ns.T >= 1
    assert ns.train_batch_size >= 1
    assert ns.patch_size >= 16
    assert ns.vlif_backend in ("torch", "cupy")
    assert ns.partition_mode in ("iid", "dirichlet_cluster")
    assert ns.eval_mode in ("fullimage", "sliding")
    assert 0.0 < ns.lr and ns.min_lr >= 0


# ---------------------------------------------------------------------------
# Optional CLI layer
# ---------------------------------------------------------------------------

def parse_v1_cli(argv: Optional[list] = None) -> argparse.Namespace:
    """Minimal CLI wrapper over build_v1_args — only the fields we expect
    to override on the command line are exposed.  Add more later as
    needed.
    """
    p = argparse.ArgumentParser(description="Run v1 cloud-removal smoke")
    p.add_argument("--data_root",    type=str,   default=V1_DEFAULTS["data_root"])
    p.add_argument("--dataset_name", type=str,   default=V1_DEFAULTS["dataset_name"])
    p.add_argument("--output_dir",   type=str,   default=V1_DEFAULTS["output_dir"])
    p.add_argument("--run_name",     type=str,   default=V1_DEFAULTS["run_name"])
    p.add_argument("--num_epoch",    type=int,   default=V1_DEFAULTS["num_epoch"])
    p.add_argument("--num_planes",   type=int,   default=V1_DEFAULTS["num_planes"])
    p.add_argument("--sats_per_plane", type=int, default=V1_DEFAULTS["sats_per_plane"])
    p.add_argument("--patch_size",   type=int,   default=V1_DEFAULTS["patch_size"])
    p.add_argument("--train_batch_size", type=int, default=V1_DEFAULTS["train_batch_size"])
    p.add_argument("--vlif_dim",     type=int,   default=V1_DEFAULTS["vlif_dim"])
    p.add_argument("--vlif_backend", type=str,   default=V1_DEFAULTS["vlif_backend"],
                   choices=["torch", "cupy"])
    p.add_argument("--lr",           type=float, default=V1_DEFAULTS["lr"])
    p.add_argument("--bn_local",     action="store_true",
                   help="Enable FedBN-style BN-local aggregation (v2 toggle).")
    p.add_argument("--eval_mode",    type=str,   default=V1_DEFAULTS["eval_mode"],
                   choices=["fullimage", "sliding"])
    p.add_argument("--seed",         type=int,   default=V1_DEFAULTS["seed"])
    p.add_argument("--device",       type=str,   default=V1_DEFAULTS["device"])
    parsed = p.parse_args(argv)
    return build_v1_args(**vars(parsed))


if __name__ == "__main__":
    ns = build_v1_args()
    print("Default v1 config:")
    for k, v in sorted(vars(ns).items()):
        print(f"  {k:22s} = {v}")
