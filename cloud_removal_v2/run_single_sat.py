"""
v2 single-satellite local-training baseline.

Simulates the "no federation" setting: pick ONE satellite from the same
v2-A Dirichlet partition that the FL sweep uses, then train a single
model centrally on only that satellite's local subset (no aggregation,
no neighbour communication).  Evaluated on the SAME global test set on
the SAME eval schedule as run_smoke.py, so the resulting curve drops
into existing v2 plots as an additional baseline alongside the
{FedAvg,FedBN} x {RelaySum,Gossip,All-Reduce} cells.

Selection rule
--------------
The 5 x 10 partition is materialised with the same partition_seed; per
satellite sample-count is computed; the satellite whose count is the
*median* is picked (tie-break: lowest flat index).  This avoids the
extreme-tail clients from Dirichlet(alpha=0.1) and gives a single
representative "typical" client.

Output
------
    Outputs_v2/v2a_<run_name>_single_sat.npz
        epochs, train_loss, eval_psnr, eval_ssim, comm_bytes (all 0),
        per_plane_psnr, per_plane_ssim, wall_seconds.
    Outputs_v2/ckpts/<run_name>_single_sat.pt
        final state_dict.
    Outputs_v2/v2a_<run_name>_summary.json
        merged 'single_sat' cell entry alongside any FL cells.

Implementation reuses run_smoke's data path and uses a 1-plane x 1-sat
CloudRemovalConstellation with ALLREDUCE (which is a no-op when
num_planes == sats_per_plane == 1).  This keeps the optimiser /
warmup / loss / eval primitives bit-identical to the FL cells.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    _this = Path(__file__).resolve()
    _parent = _this.parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v2.run_single_sat import main  # noqa: E402
    if __name__ == "__main__":
        main()
        sys.exit(0)

from .config import parse_v2a_cli
from .dataset import (
    build_plane_satellite_partitions_v2,
    seed_worker,
)
from .run_smoke import (
    _atomic_savez,
    _atomic_save_torch,
    _atomic_write_json,
    _ensure_omp_threads,
    _load_datasets,
    _log,
    _negotiate_backend,
    _set_seed,
)

from cloud_removal_v1.constants import ALLREDUCE
from cloud_removal_v1.constellation import CloudRemovalConstellation
from cloud_removal_v1.evaluation import evaluate_per_plane, average_eval_results


def _pick_median_satellite(per_client_sizes: List[List[int]]) -> Tuple[int, int, int]:
    """Return (plane, sat, size) of the satellite with the median sample
    count (tie-break: lowest flat index = plane * sats_per_plane + sat)."""
    flat: List[Tuple[int, int, int]] = []
    for p, row in enumerate(per_client_sizes):
        for s, n in enumerate(row):
            flat.append((p, s, int(n)))
    sorted_by_size = sorted(flat, key=lambda t: (t[2], t[0] * len(per_client_sizes[0]) + t[1]))
    median = sorted_by_size[len(sorted_by_size) // 2]
    return median


def main(argv=None) -> None:
    args = parse_v2a_cli(argv)
    _ensure_omp_threads()
    _set_seed(args.seed, deterministic=getattr(args, "deterministic", False))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"device: {device}")
    if device.type == "cuda":
        _log(f"CUDA device name: {torch.cuda.get_device_name(device)}")
    args.vlif_backend = _negotiate_backend(args.vlif_backend)

    # -- Data: same partition as FL sweep ----------------------------------
    train_ms, test_ms = _load_datasets(args)

    aug_params = {
        "hflip_p":  args.aug_hflip_p,
        "vflip_p":  args.aug_vflip_p,
        "rot90_p":  args.aug_rot90_p,
        "rot270_p": args.aug_rot270_p,
    }
    client_datasets = build_plane_satellite_partitions_v2(
        train_ms,
        num_planes=args.num_planes,
        sats_per_plane=args.sats_per_plane,
        mode=args.partition_mode,
        alpha=args.partition_alpha,
        seed=args.partition_seed,
        min_per_client=args.min_samples_per_client,
        augment=bool(args.augment),
        augment_params=aug_params,
    )
    per_client_sizes = [[len(ds) for ds in row] for row in client_datasets]
    _log(f"per-client sizes: {per_client_sizes}")

    p_sel, s_sel, n_sel = _pick_median_satellite(per_client_sizes)
    _log(f"single-sat baseline: picked plane={p_sel} sat={s_sel} "
         f"with {n_sel} samples (median of {args.num_planes * args.sats_per_plane})")

    # Wrap chosen satellite as a 1x1 constellation grid.  ALLREDUCE on
    # a single plane is a no-op aggregation, and intra-plane average
    # over a single sat is the identity, so train_one_round becomes
    # pure local training -- exactly what we want for "no federation".
    chosen_dataset = client_datasets[p_sel][s_sel]
    grid = [[chosen_dataset]]

    # Same test loader as run_smoke
    test_loader_kwargs: Dict = dict(
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        test_loader_kwargs["worker_init_fn"] = seed_worker
    test_loader = DataLoader(test_ms, **test_loader_kwargs)

    # Hijack args for the 1x1 constellation, restore on exit so the
    # config snapshot we write to summary.json reflects the ORIGINAL
    # 5x10 sweep settings (paired-comparison context).
    orig_num_planes = args.num_planes
    orig_sats_per_plane = args.sats_per_plane
    args.num_planes = 1
    args.sats_per_plane = 1

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    try:
        constellation = CloudRemovalConstellation(
            num_planes=1,
            sats_per_plane=1,
            client_datasets=grid,
            args=args,
            init_state_dict=None,
            device=str(device),
            logger=_log,
        )

        history = {
            "epochs":         [],
            "train_loss":     [],
            "eval_psnr":      [],
            "eval_ssim":      [],
            "comm_bytes":     [],
            "per_plane_psnr": [],
            "per_plane_ssim": [],
            "wall_seconds":   [],
        }

        for ep in range(1, args.num_epoch + 1):
            t0 = time.time()
            train_loss = constellation.train_one_round(ALLREDUCE)
            if not (train_loss == train_loss):  # NaN check
                raise RuntimeError(
                    f"single_sat ep {ep}: training loss is NaN. "
                    f"Lower --lr or check the chosen client subset."
                )

            do_eval = (ep % args.eval_every == 0) or (ep == args.num_epoch)
            if do_eval:
                try:
                    per_plane = evaluate_per_plane(
                        constellation, test_loader,
                        mode=args.eval_mode,
                        patch_size=args.eval_patch_size,
                        window=args.sliding_window,
                        stride=args.sliding_stride,
                        device=device,
                    )
                    mean = average_eval_results(per_plane)
                    psnr, ssim = mean.mean_psnr, mean.mean_ssim
                    pp_psnr = [r.mean_psnr for r in per_plane]
                    pp_ssim = [r.mean_ssim for r in per_plane]
                except Exception as e:
                    _log(f"WARN: eval failed at ep {ep} "
                         f"({type(e).__name__}: {e}); recording NaN")
                    psnr, ssim, pp_psnr, pp_ssim = np.nan, np.nan, [], []
            else:
                psnr, ssim, pp_psnr, pp_ssim = np.nan, np.nan, [], []

            dt = time.time() - t0
            history["epochs"].append(ep)
            history["train_loss"].append(float(train_loss))
            history["eval_psnr"].append(float(psnr))
            history["eval_ssim"].append(float(ssim))
            # No federation -> no inter-satellite communication.
            history["comm_bytes"].append(0)
            history["per_plane_psnr"].append(pp_psnr)
            history["per_plane_ssim"].append(pp_ssim)
            history["wall_seconds"].append(dt)

            _log(f"[single_sat] ep {ep:02d}/{args.num_epoch}  "
                 f"loss={train_loss:.4f}  PSNR={psnr:.3f}dB  "
                 f"SSIM={ssim:.4f}  time={dt:.1f}s")

        final_sd = constellation.planes[0][0].get_weights(cpu=True)
        del constellation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        args.num_planes = orig_num_planes
        args.sats_per_plane = orig_sats_per_plane

    tag = f"{args.run_name}_single_sat"
    npz_path = os.path.join(args.output_dir, f"v2a_{tag}.npz")
    _atomic_savez(
        npz_path,
        epochs=np.array(history["epochs"]),
        train_loss=np.array(history["train_loss"]),
        eval_psnr=np.array(history["eval_psnr"]),
        eval_ssim=np.array(history["eval_ssim"]),
        comm_bytes=np.array(history["comm_bytes"]),
        wall_seconds=np.array(history["wall_seconds"]),
        per_plane_psnr=np.array(history["per_plane_psnr"], dtype=object),
        per_plane_ssim=np.array(history["per_plane_ssim"], dtype=object),
        picked_plane=np.array(p_sel),
        picked_sat=np.array(s_sel),
        picked_size=np.array(n_sel),
    )
    _log(f"wrote {npz_path}")

    ckpt_path = os.path.join(args.ckpt_dir, f"{tag}.pt")
    _atomic_save_torch(final_sd, ckpt_path)
    _log(f"wrote {ckpt_path}")

    # Merge into the same summary.json the FL sweep populates.
    summary_path = os.path.join(args.output_dir,
                                f"v2a_{args.run_name}_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            if "final" not in summary or not isinstance(summary["final"], dict):
                summary["final"] = {}
        except Exception:
            summary = {"final": {}}
    else:
        summary = {"final": {}}

    ep_arr = np.asarray(history["epochs"])
    psnr_arr = np.asarray(history["eval_psnr"], dtype=float)
    ssim_arr = np.asarray(history["eval_ssim"], dtype=float)
    valid = ~np.isnan(psnr_arr)
    if valid.any():
        last_valid = int(ep_arr[valid][-1])
        psnr_last = float(psnr_arr[valid][-1])
        ssim_last = float(ssim_arr[valid][-1])
    else:
        last_valid, psnr_last, ssim_last = -1, float("nan"), float("nan")
    summary["final"]["single_sat"] = {
        "PSNR_final": psnr_last,
        "SSIM_final": ssim_last,
        "final_eval_epoch": last_valid,
        "train_loss_final": float(history["train_loss"][-1]),
        "total_comm_bytes": 0,
        "total_wall_seconds": float(sum(history["wall_seconds"])),
        "picked_plane": int(p_sel),
        "picked_sat": int(s_sel),
        "picked_size": int(n_sel),
    }
    summary["config"] = {k: v for k, v in vars(args).items()
                         if isinstance(v, (int, float, str, bool, list, tuple))}
    _atomic_write_json(summary_path, summary)
    _log(f"merged single_sat into {summary_path}")
    _log("done.")


if __name__ == "__main__":
    main()
