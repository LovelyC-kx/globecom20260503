"""
Generate v1 comparison plots from the .npz dumps produced by
`run_v1_smoke.py`.

Outputs three figures (styled to match the original FLSNN Fig 5):
    Outputs/v1_<run>_train_loss.pdf
    Outputs/v1_<run>_test_psnr.pdf
    Outputs/v1_<run>_test_ssim.pdf

One curve per aggregation scheme (RelaySum / Gossip / AllReduce),
markers + line colours replicated from the original `constants.py`
colour palette.

Usage
-----
    python plot_v1_results.py --run_name v1_smoke
    python plot_v1_results.py --run_name v1_smoke --output_dir ./Outputs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Keep compatibility with the original plotting conventions in the
# classification codebase (main.py, aggregation_comparison.py etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import color_list, marker_list, GOSSIP, RELAYSUM, ALLREDUCE


_SCHEME_ORDER = [RELAYSUM, GOSSIP, ALLREDUCE]
_LEGEND_LABEL = {
    RELAYSUM:  "RelaySum (Proposed)",
    GOSSIP:    "Gossip",
    ALLREDUCE: "All-Reduce",
}


def _load_scheme(output_dir: Path, run_name: str, scheme: str) -> Optional[Dict]:
    tag = scheme.replace(" ", "_")
    path = output_dir / f"v1_smoke_{run_name}_{tag}.npz"
    if not path.exists():
        print(f"WARN: missing {path}", file=sys.stderr)
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "epochs":       data["epochs"],
        "train_loss":   data["train_loss"],
        "eval_psnr":    data["eval_psnr"],
        "eval_ssim":    data["eval_ssim"],
        "comm_bytes":   data["comm_bytes"],
    }


def _plot_metric(series: Dict[str, Dict],
                 metric: str,
                 ylabel: str,
                 output_path: Path,
                 percent: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    fig = plt.figure(figsize=(10, 8))
    line_list = []
    legends = []

    for idx, scheme in enumerate(_SCHEME_ORDER):
        if scheme not in series or series[scheme] is None:
            continue
        s = series[scheme]
        xs = s["epochs"]
        ys = s[metric]
        mask = ~np.isnan(ys) if np.issubdtype(ys.dtype, np.floating) else slice(None)
        if np.issubdtype(ys.dtype, np.floating):
            xs_m = xs[mask]
            ys_m = ys[mask]
        else:
            xs_m = xs
            ys_m = ys
        if len(xs_m) == 0:
            continue
        line, = plt.plot(
            xs_m, ys_m,
            color=color_list[idx % len(color_list)],
            linestyle="-",
            marker=marker_list[idx % len(marker_list)],
            markerfacecolor="none", ms=7,
            markeredgewidth=2.5, linewidth=2.5,
            markevery=max(1, len(xs_m) // 10),
        )
        line_list.append(line)
        legends.append(_LEGEND_LABEL[scheme])

    plt.legend(line_list, legends, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Global Epoch", fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    if percent:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()
    plt.grid(True, alpha=0.5)

    fig.savefig(output_path, format="pdf", dpi=1200)
    plt.close(fig)
    print(f"wrote {output_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name",   type=str, default="v1_smoke")
    p.add_argument("--output_dir", type=str, default="./Outputs")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)

    series: Dict[str, Dict] = {
        scheme: _load_scheme(out_dir, args.run_name, scheme)
        for scheme in _SCHEME_ORDER
    }

    any_data = any(v is not None for v in series.values())
    if not any_data:
        print("No .npz results found — did you run run_v1_smoke.py first?",
              file=sys.stderr)
        sys.exit(1)

    _plot_metric(series, "train_loss", "Training Loss",
                 out_dir / f"v1_{args.run_name}_train_loss.pdf")
    _plot_metric(series, "eval_psnr",  "Test PSNR (dB)",
                 out_dir / f"v1_{args.run_name}_test_psnr.pdf")
    _plot_metric(series, "eval_ssim",  "Test SSIM",
                 out_dir / f"v1_{args.run_name}_test_ssim.pdf")


if __name__ == "__main__":
    main()
