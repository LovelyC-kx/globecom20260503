"""
v1 smoke run — three-scheme inter-plane comparison on CUHK-CR1.

Reproduces the spirit of the original FLSNN Fig 5 (Inter-Plane Aggregation
Comparison), but for:
  * image regression  (cloud removal, not land-cover classification),
  * VLIFNet backbone  (not SpikingCNN / SmallResNet),
  * RGB paired CUHK-CR1 data (not EuroSAT pkl),
  * 50/5/1 Walker Star (num_planes=5, sats_per_plane=10).

Output
------
Outputs/
    v1_smoke_<run_name>_<scheme>.npz
        per-epoch train_loss, PSNR, SSIM, comm_bytes, per-plane breakdown.
    v1_smoke_<run_name>_summary.json
        final numbers + config snapshot.
    tb/<run_name>/                                (tensorboard logs)

Usage
-----
    python run_v1_smoke.py                                      # all defaults
    python run_v1_smoke.py --data_root ./data/CUHK-CR1 \\
        --num_epoch 10 --run_name my_exp
    python run_v1_smoke.py --eval_mode sliding                  # R-v1-7 OOM fallback

The script is fully deterministic given `--seed`.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make sibling modules importable when running from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constants import GOSSIP, RELAYSUM, ALLREDUCE
from cloud_removal_config import parse_v1_cli
from cloud_removal_dataset import (
    PairedCloudDataset,
    split_train_test,
    build_plane_satellite_partitions,
)
from cloud_removal_constellation import CloudRemovalConstellation
from cloud_removal_eval import (
    evaluate_fullimage,
    evaluate_sliding,
    evaluate_per_plane,
    average_eval_results,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(msg: str) -> None:
    print(f"[v1] {msg}", flush=True)


def _load_datasets(args) -> Tuple[PairedCloudDataset,
                                  "torch.utils.data.Dataset",  # noqa: F821
                                  "torch.utils.data.Dataset"]:
    """Return (base, train, test) where train/test are either direct
    PairedCloudDataset views or Subsets derived from an 8:2 random split
    (if the root has no explicit train/test sub-folders)."""
    # Attempt explicit train/test split first
    try:
        train_ds = PairedCloudDataset(args.data_root,
                                      split=args.train_split,
                                      patch_size=args.patch_size)
        test_ds  = PairedCloudDataset(args.data_root,
                                      split=args.test_split,
                                      patch_size=None)
        _log(f"Explicit split: train={len(train_ds)}  test={len(test_ds)}")
        return train_ds, train_ds, test_ds
    except FileNotFoundError:
        pass

    base = PairedCloudDataset(args.data_root, split=None,
                              patch_size=args.patch_size)
    _log(f"Flat layout: {base.describe()}")
    train_sub, test_sub = split_train_test(base, test_ratio=0.2,
                                           seed=args.partition_seed)
    _log(f"Derived 8:2 split: |train|={len(train_sub)}  |test|={len(test_sub)}")
    return base, train_sub, test_sub


def _cupy_available() -> bool:
    try:
        import cupy   # noqa: F401
        return True
    except Exception:
        return False


def _negotiate_backend(requested: str) -> str:
    if requested == "cupy" and not _cupy_available():
        _log("WARNING: cupy not importable; falling back to backend='torch'")
        return "torch"
    return requested


# ---------------------------------------------------------------------------
# One scheme
# ---------------------------------------------------------------------------

def _run_scheme(scheme: str,
                args,
                test_loader: DataLoader,
                client_datasets,
                device: torch.device,
                writer,
                ) -> Dict:
    """Train+evaluate one aggregation scheme for args.num_epoch rounds.

    Returns a dict ready for .npz + JSON serialisation.
    """
    _log(f"===== scheme = {scheme} =====")

    constellation = CloudRemovalConstellation(
        num_planes=args.num_planes,
        sats_per_plane=args.sats_per_plane,
        client_datasets=client_datasets,
        args=args,
        device=str(device),
        logger=_log,
    )

    history = {
        "epochs":          [],
        "train_loss":      [],
        "eval_psnr":       [],
        "eval_ssim":       [],
        "comm_bytes":      [],
        "per_plane_psnr":  [],
        "per_plane_ssim":  [],
        "wall_seconds":    [],
    }

    for ep in range(1, args.num_epoch + 1):
        t0 = time.time()
        train_loss = constellation.train_one_round(scheme)
        # Evaluation: every `eval_every` epochs or on the last round
        if ep % args.eval_every == 0 or ep == args.num_epoch:
            if args.eval_mode == "sliding":
                # evaluate_per_plane internally builds a batch_size=1 loader
                per_plane = evaluate_per_plane(
                    constellation, test_loader,
                    window=args.sliding_window,
                    stride=args.sliding_stride,
                    device=device,
                )
            else:
                per_plane = evaluate_per_plane(
                    constellation, test_loader,
                    window=0, stride=0, device=device,
                )
            mean = average_eval_results(per_plane)
            psnr = mean.mean_psnr
            ssim = mean.mean_ssim
            pp_psnr = [r.mean_psnr for r in per_plane]
            pp_ssim = [r.mean_ssim for r in per_plane]
        else:
            psnr, ssim, pp_psnr, pp_ssim = np.nan, np.nan, [], []

        dt = time.time() - t0
        cb = int(constellation.round_bytes[-1])

        history["epochs"].append(ep)
        history["train_loss"].append(float(train_loss))
        history["eval_psnr"].append(float(psnr))
        history["eval_ssim"].append(float(ssim))
        history["comm_bytes"].append(cb)
        history["per_plane_psnr"].append(pp_psnr)
        history["per_plane_ssim"].append(pp_ssim)
        history["wall_seconds"].append(dt)

        _log(f"[{scheme}] ep {ep:02d}/{args.num_epoch}  "
             f"loss={train_loss:.4f}  "
             f"PSNR={psnr:.3f}dB  SSIM={ssim:.4f}  "
             f"comm={cb/1e6:.1f}MB  "
             f"time={dt:.1f}s")

        if writer is not None:
            writer.add_scalar(f"{scheme}/train_loss", train_loss, ep)
            if not np.isnan(psnr):
                writer.add_scalar(f"{scheme}/eval_psnr",  psnr, ep)
                writer.add_scalar(f"{scheme}/eval_ssim",  ssim, ep)
            writer.add_scalar(f"{scheme}/comm_bytes", cb, ep)

    # Release the model memory before the next scheme spins up 50 new nets.
    del constellation
    torch.cuda.empty_cache()
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_v1_cli(argv)
    _set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")
    if device.type == "cuda":
        _log(f"CUDA device name: {torch.cuda.get_device_name(device)}")

    args.vlif_backend = _negotiate_backend(args.vlif_backend)
    _log(f"VLIFNet backend: {args.vlif_backend}")

    # -- Data --------------------------------------------------------------
    _, train_sub, test_sub = _load_datasets(args)

    total_clients = args.num_planes * args.sats_per_plane
    assert len(train_sub) >= total_clients, (
        f"Train set too small ({len(train_sub)}) for "
        f"{args.num_planes}×{args.sats_per_plane}={total_clients} clients.")

    client_datasets = build_plane_satellite_partitions(
        train_sub,
        num_planes=args.num_planes,
        sats_per_plane=args.sats_per_plane,
        mode=args.partition_mode,
        seed=args.partition_seed,
    )
    per_client_sizes = [[len(ds) for ds in row] for row in client_datasets]
    _log(f"per-client dataset sizes: {per_client_sizes}")

    test_loader = DataLoader(test_sub, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # -- Tensorboard -------------------------------------------------------
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, args.run_name))
    except Exception as e:
        _log(f"tensorboard disabled ({e})")

    # -- Run three schemes -------------------------------------------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Dict] = {}
    for scheme in (RELAYSUM, GOSSIP, ALLREDUCE):
        scheme_tag = scheme.replace(" ", "_")
        out_npz = os.path.join(
            args.output_dir,
            f"v1_smoke_{args.run_name}_{scheme_tag}.npz")

        history = _run_scheme(scheme, args, test_loader, client_datasets,
                              device, writer)
        np.savez(out_npz,
                 epochs=np.array(history["epochs"]),
                 train_loss=np.array(history["train_loss"]),
                 eval_psnr=np.array(history["eval_psnr"]),
                 eval_ssim=np.array(history["eval_ssim"]),
                 comm_bytes=np.array(history["comm_bytes"]),
                 wall_seconds=np.array(history["wall_seconds"]),
                 per_plane_psnr=np.array(history["per_plane_psnr"], dtype=object),
                 per_plane_ssim=np.array(history["per_plane_ssim"], dtype=object))
        _log(f"wrote {out_npz}")
        all_results[scheme] = history

    if writer is not None:
        writer.close()

    # -- Summary JSON ------------------------------------------------------
    summary = {
        "config": {k: v for k, v in vars(args).items()
                   if isinstance(v, (int, float, str, bool, list, tuple))},
        "final": {
            scheme: {
                "PSNR": float(all_results[scheme]["eval_psnr"][-1]),
                "SSIM": float(all_results[scheme]["eval_ssim"][-1]),
                "train_loss": float(all_results[scheme]["train_loss"][-1]),
                "total_comm_bytes": int(sum(all_results[scheme]["comm_bytes"])),
                "total_wall_seconds": float(sum(all_results[scheme]["wall_seconds"])),
            }
            for scheme in all_results
        },
    }
    summary_path = os.path.join(args.output_dir,
                                f"v1_smoke_{args.run_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"wrote {summary_path}")
    _log("done.")


if __name__ == "__main__":
    main()
