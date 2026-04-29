"""
plot_paper_figs.py — top-tier-journal-quality figures for the OrbitVLIF paper.

Generates 5 PDF figures + 3 LaTeX tables from training outputs.

Figures (β scheme; Fig 1 architecture is hand-drawn TikZ, not produced here):
    fig2_centralized_curves.pdf  : A1 / A2 / C2 PSNR-vs-epoch (CR1, CR2 panels)
    fig3_qualitative_grid.pdf    : 4 test imgs × 4 cols (Cloudy, VLIF, Plain, GT)
    fig4_ablation_bars.pdf       : A1 vs B1 / B2 / B3 (PSNR, SSIM dual subplot)
    fig5_federated_curves.pdf    : F_SNN / F_ANN / F_plain PSNR-vs-round
                                    + per-plane shading + comm-bytes inset
    fig6_energy_bars.pdf         : ANN_pJ / SNN_lower / SNN_upper, log scale

Tables:
    tab1_centralized.tex
    tab2_ablation.tex
    tab3_federated.tex

Style:  STIXGeneral font, color-blind-safe 4-palette, IEEE 2-col conf sizes,
        booktabs LaTeX, 600-dpi vector PDF.

Usage
-----
    python -m cloud_removal_v1.plot_paper_figs \\
        --outputs_v1 ./Outputs_v1 --outputs_v2 ./Outputs_v2 \\
        --energy_dir ./Outputs_energy_A1 \\
        --out_dir ./figures \\
        --figs all --tables yes

    # Re-generate only Fig 5
    python -m cloud_removal_v1.plot_paper_figs --figs 5 --tables no

Missing input files are SKIPPED with a WARNING (never error) so the script
can be re-run during long training to refresh partial figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow `python cloud_removal_v1/plot_paper_figs.py` as well as the -m form.
if __package__ in (None, ""):
    _parent = Path(__file__).resolve().parent.parent
    if str(_parent) not in sys.path:
        sys.path.insert(0, str(_parent))
    from cloud_removal_v1.plot_paper_figs import main  # noqa: E402
    if __name__ == "__main__":
        main()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Global style — color-blind-safe 4-palette + IEEE conference sizes
# ---------------------------------------------------------------------------

# Wong-2011 color-blind-safe; verified against deuteranopia / protanopia sims.
PALETTE_BLUE   = "#0173B2"   # primary    — VLIFNet / SNN
PALETTE_ORANGE = "#DE8F05"   # secondary  — PlainUNet / ANN
PALETTE_GREEN  = "#029E73"   # tertiary   — ESDNet / extra
PALETTE_GRAY   = "#999999"   # neutral    — ground truth / shading

# IEEE 2-column conference, sizes in inches.
FIG_SINGLE_COL = (3.5, 2.4)
FIG_DOUBLE_COL = (7.2, 2.6)
FIG_GRID       = (7.2, 7.0)   # qualitative grid

# Markers used at every Nth epoch / round.
MARKERS = ("o", "s", "D", "^", "v")

# Marker-every-N — set so each curve has ~6–8 markers regardless of length.
MARKER_TARGET_COUNT = 7


# ---------------------------------------------------------------------------
# Default run-name registry — overridable via CLI.
# These match the canonical commands shipped in the README / paper notes.
# ---------------------------------------------------------------------------

_DEFAULT_RUNS = {
    "a1":         "A1_vlif_cr1",
    "a2":         "A2_vlif_cr2",
    "c2_cr1":     "C2_plain_ann_cr1",
    "c2_cr2":     "C2_plain_ann_cr2",
    "b1":         "B1_no_fsta_cr1",
    "b2":         "B2_no_dual_group_cr1",
    "b3":         "B3_binary_spike_cr1",
    "f_snn":      "F_snn",       # → v2a_F_snn_fedbn_Gossip_Averaging.npz
    "f_ann":      "F_ann",
    "f_plain":    "F_plain",
}

# Canonical (BN, scheme) cell — locked at fedbn × Gossip per paper decision.
F_BN     = "fedbn"
F_SCHEME = "Gossip_Averaging"


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate paper figures + LaTeX tables from training outputs.")

    # Input / output paths
    p.add_argument("--outputs_v1", type=str, default="./Outputs_v1",
                   help="Directory containing centralized_*.npz / *_best.pt / "
                        "*_summary.json (output of train_centralized.py).")
    p.add_argument("--outputs_v2", type=str, default="./Outputs_v2",
                   help="Directory containing v2a_*.npz / ckpts/ / "
                        "v2a_*_summary.json (output of run_smoke.py).")
    p.add_argument("--energy_dir", type=str, default="./Outputs_energy_A1",
                   help="Directory containing energy_summary.json "
                        "(output of energy_estimation.py).  Required for Fig 6.")
    p.add_argument("--out_dir", type=str, default="./figures",
                   help="Where PDFs and .tex files are written.")

    # What to render
    p.add_argument("--figs", type=str, default="all",
                   help="'all' or comma-list, e.g. '2,3,5'.")
    p.add_argument("--tables", type=str, default="yes", choices=["yes", "no"])

    # Run-name overrides (defaults match _DEFAULT_RUNS).
    for key, default in _DEFAULT_RUNS.items():
        p.add_argument(f"--run_{key}", type=str, default=default)

    # Inference settings (Fig 3)
    p.add_argument("--qual_n_samples", type=int, default=4,
                   help="Number of qualitative-grid rows (test images).")
    p.add_argument("--qual_seed", type=int, default=0,
                   help="Deterministic test-image selection seed.")
    p.add_argument("--qual_dataset_root", type=str, default=None,
                   help="Test-set root for Fig 3.  Defaults to the data_root "
                        "stored in the A1 ckpt's saved config.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for Fig 3 inference (CPU is fine, ~4 imgs).")

    # Federated cell selection (paper-locked default = fedbn × Gossip)
    p.add_argument("--f_bn", type=str, default=F_BN, choices=["fedavg", "fedbn"])
    p.add_argument("--f_scheme", type=str, default=F_SCHEME)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging — _warn skips a missing artefact instead of aborting the whole run
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[plot_paper_figs] {msg}", flush=True)


def _warn(msg: str) -> None:
    """Skip-with-warning: missing files are common during in-flight training."""
    warnings.warn(f"[plot_paper_figs] {msg}", stacklevel=2)


def _resolve_figs(spec: str) -> List[int]:
    """'all' | '2,3,5' | '4'  →  sorted list of int fig ids in {2,3,4,5,6}."""
    valid = {2, 3, 4, 5, 6}
    if spec.strip().lower() == "all":
        return sorted(valid)
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"--figs token {tok!r} not an int")
        i = int(tok)
        if i not in valid:
            raise ValueError(f"--figs id {i} not in {sorted(valid)}")
        out.append(i)
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Block 2 onwards (data loaders, style helpers, fig fns, table fns, main)
# will be appended by subsequent commits — see plan in chat.
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    """Stub — will be filled in once all blocks are in place."""
    args = _parse_args(argv)
    fig_ids = _resolve_figs(args.figs)
    _log(f"out_dir={args.out_dir}  figs={fig_ids}  tables={args.tables}")
    _log("plot_paper_figs.py: skeleton only — fig/table generators not yet wired.")


if __name__ == "__main__":
    main()
