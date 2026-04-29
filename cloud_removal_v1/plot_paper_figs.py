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
# Block 2 — data loaders (pure IO; missing files → None + _warn, never raise)
# ---------------------------------------------------------------------------

def load_centralized_npz(outputs_v1: Path,
                         run_name: str) -> Optional[Dict[str, np.ndarray]]:
    """Read ``Outputs_v1/centralized_<run>.npz`` written by train_centralized.py.

    Returns a dict with keys::

        epoch, lr, train_loss, train_charbonnier, train_ssim_loss,
        eval_psnr, eval_ssim, wall_seconds

    or None if the file is missing.  Each value is a 1-D ``np.ndarray``.
    """
    path = outputs_v1 / f"centralized_{run_name}.npz"
    if not path.is_file():
        _warn(f"missing centralized npz: {path}")
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k].copy() for k in d.files}


def load_centralized_summary(outputs_v1: Path,
                             run_name: str) -> Optional[Dict]:
    """Read ``Outputs_v1/centralized_<run>_summary.json``.

    Returns the parsed dict (config / params_M / best / final / total_wall_seconds)
    or None if missing.
    """
    path = outputs_v1 / f"centralized_{run_name}_summary.json"
    if not path.is_file():
        _warn(f"missing centralized summary: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_federated_npz(outputs_v2: Path,
                       run_name: str,
                       bn_mode: str = F_BN,
                       scheme: str = F_SCHEME,
                       ) -> Optional[Dict[str, np.ndarray]]:
    """Read ``Outputs_v2/v2a_<run>_<bn>_<scheme>.npz`` from run_smoke.py.

    Returns a dict with keys::

        epochs, train_loss, eval_psnr, eval_ssim, comm_bytes,
        per_plane_psnr, per_plane_ssim, wall_seconds

    where ``per_plane_psnr`` / ``per_plane_ssim`` are object-dtype arrays
    (list of per-plane scalar lists), so we load with ``allow_pickle=True``.
    None if the file is missing.
    """
    path = outputs_v2 / f"v2a_{run_name}_{bn_mode}_{scheme}.npz"
    if not path.is_file():
        _warn(f"missing federated npz: {path}")
        return None
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k].copy() for k in d.files}


def load_federated_summary(outputs_v2: Path,
                           run_name: str) -> Optional[Dict]:
    """Read ``Outputs_v2/v2a_<run>_summary.json`` (config + final per-cell numbers)."""
    path = outputs_v2 / f"v2a_{run_name}_summary.json"
    if not path.is_file():
        _warn(f"missing federated summary: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_energy_summary(energy_dir: Path) -> Optional[Dict]:
    """Read ``energy_summary.json`` from energy_estimation.py.

    Returns None if missing or malformed.  The dict has::

        ckpt, config, params_M, n_samples_evaluated,
        energy_per_image: {ann_macs, snn_effective_acs,
                           energy_ann_pj, energy_snn_lower_pj,
                           energy_snn_upper_pj},
        per_layer_macs:  [...],
        per_layer_spikes:[...],
    """
    path = energy_dir / "energy_summary.json"
    if not path.is_file():
        _warn(f"missing energy summary: {path}")
        return None
    with open(path, "r") as f:
        blob = json.load(f)
    if "energy_per_image" not in blob:
        _warn(f"energy summary {path} missing 'energy_per_image' key")
        return None
    return blob


def load_ckpt_for_inference(ckpt_path: Path,
                            device: "torch.device",
                            ) -> Optional[Tuple["nn.Module", Dict, bool]]:
    """Reconstruct model + load weights from a centralized ``*_best.pt`` ckpt.

    Returns (model.eval(), cfg, is_snn) on success, or None on miss / failure.

    Delegates to :func:`cloud_removal_v2.energy_estimation._build_model_from_ckpt`
    which already handles backbone dispatch (vlif / vlif_ann / plain_ann),
    ablation re-application, and strict=False weight loading.
    """
    if not Path(ckpt_path).is_file():
        _warn(f"missing ckpt: {ckpt_path}")
        return None
    try:
        from cloud_removal_v2.energy_estimation import _build_model_from_ckpt
        model, cfg = _build_model_from_ckpt(str(ckpt_path), device)
    except Exception as e:                    # noqa: BLE001
        _warn(f"failed to load {ckpt_path}: {type(e).__name__}: {e}")
        return None
    is_snn = cfg.get("backbone", "vlif") == "vlif"
    return model, cfg, is_snn


# ---------------------------------------------------------------------------
# Block 3 — global style helpers + Fig 2 (centralized training curves)
# ---------------------------------------------------------------------------

_MPL_INITIALISED = False


def _setup_mpl() -> None:
    """Apply global matplotlib rcParams once.  Idempotent — repeat calls
    are cheap, so figure functions can call this freely on entry.

    Style targets IEEE 2-column conference proceedings:
      * STIXGeneral font (Times-like, math-compatible)
      * Vector PDF backend (Agg buffer)
      * 0.6 pt axes / 0.5 pt grid / 1.4 pt curves / 5.5 pt markers
      * 8 pt tick labels, 9 pt axis labels, 10 pt panel titles
      * legend frameon=False, fancybox=False
    """
    global _MPL_INITIALISED
    if _MPL_INITIALISED:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":         "STIXGeneral",
        "mathtext.fontset":    "stix",
        "pdf.fonttype":        42,        # TrueType fonts (editable in Illustrator)
        "ps.fonttype":         42,
        "axes.linewidth":      0.6,
        "axes.titlesize":      10,
        "axes.labelsize":      9,
        "xtick.labelsize":     8,
        "ytick.labelsize":     8,
        "xtick.major.width":   0.6,
        "ytick.major.width":   0.6,
        "xtick.major.size":    3.0,
        "ytick.major.size":    3.0,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "legend.fontsize":     8,
        "legend.frameon":      False,
        "legend.fancybox":     False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.9,
        "lines.linewidth":     1.4,
        "lines.markersize":    5.5,
        "lines.markeredgewidth": 0.0,
        "grid.linewidth":      0.4,
        "grid.linestyle":      "--",
        "grid.alpha":          0.4,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.02,
    })
    _MPL_INITIALISED = True


def _save_pdf(fig, out_path: Path) -> None:
    """Vector PDF write at 600 dpi (raster fallbacks crisp on print)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", dpi=600)
    _log(f"wrote {out_path}")


def _annotate_panel(ax, label: str,
                    x: float = 0.04, y: float = 0.95) -> None:
    """Add ``(a)`` / ``(b)`` panel label in axes-fraction coords (top-left)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


def _smart_marker_every(n_points: int,
                        target_count: int = MARKER_TARGET_COUNT) -> int:
    """Choose ``markevery`` so each curve has ~target markers regardless
    of the number of evaluated epochs / rounds.  Always returns >=1."""
    if n_points <= target_count:
        return 1
    return max(1, n_points // target_count)


def _finite(epoch: np.ndarray, value: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Filter (epoch, value) to indices where value is finite — train_centralized
    writes NaN on epochs where eval was skipped (eval_every > 1)."""
    mask = np.isfinite(value)
    return epoch[mask], value[mask]


# ---------------------------------------------------------------------------
# Fig 2 — centralized training curves (CR1, CR2 dual panel)
# ---------------------------------------------------------------------------

def fig2_centralized_curves(args, out_dir: Path) -> None:
    """Two side-by-side panels showing PSNR-vs-epoch on CR1 (left) and CR2
    (right).  Each panel overlays VLIFNet (blue, ``o``) and PlainUNet
    (orange, ``s``) when their npz files are present.  If a panel has
    NO data, it is annotated "no data" and the figure is still produced.

    Reads from ``args.outputs_v1`` using run-name overrides ``args.run_a1``,
    ``args.run_a2``, ``args.run_c2_cr1``, ``args.run_c2_cr2``.

    Output: ``out_dir/fig2_centralized_curves.pdf``.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    outputs_v1 = Path(args.outputs_v1)

    # Per-panel curve specs: (run_name, label, color, marker)
    panels = [
        ("CR1", [
            (args.run_a1,     "VLIFNet (ours)",  PALETTE_BLUE,   "o"),
            (args.run_c2_cr1, "PlainUNet",       PALETTE_ORANGE, "s"),
        ]),
        ("CR2", [
            (args.run_a2,     "VLIFNet (ours)",  PALETTE_BLUE,   "o"),
            (args.run_c2_cr2, "PlainUNet",       PALETTE_ORANGE, "s"),
        ]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL, sharey=False)

    n_curves_total = 0
    last_handles: List = []
    last_labels:  List[str] = []

    for ax, (panel_label, curves) in zip(axes, panels):
        ax.grid(True, which="major")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"({'a' if panel_label == 'CR1' else 'b'}) {panel_label}")

        n_drawn = 0
        for run_name, label, color, marker in curves:
            d = load_centralized_npz(outputs_v1, run_name)
            if d is None:
                continue
            ep, psnr = _finite(d["epoch"], d["eval_psnr"])
            if ep.size == 0:
                _warn(f"{run_name}: no finite eval_psnr points; skipping curve")
                continue
            mev = _smart_marker_every(ep.size)
            (ln,) = ax.plot(ep, psnr,
                            color=color, marker=marker, markevery=mev,
                            label=label, linestyle="-", clip_on=True)
            n_drawn += 1
            n_curves_total += 1
            # Capture for shared legend (latest non-empty panel wins; both
            # panels have identical curve specs anyway).
            last_handles.append(ln)
            last_labels.append(label)

        if n_drawn == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=PALETTE_GRAY, fontsize=10)

        # Auto-zoom Y-axis around the data so dB differences read cleanly.
        ax.margins(x=0.02)

    # Shared legend above the two panels (keeps panel area clean).
    if last_handles:
        # Deduplicate by label while preserving order.
        seen = set()
        uniq_h, uniq_l = [], []
        for h, l in zip(last_handles, last_labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
        fig.legend(uniq_h, uniq_l,
                   loc="upper center", bbox_to_anchor=(0.5, 1.04),
                   ncol=len(uniq_l))

    fig.tight_layout(pad=0.3)
    if n_curves_total == 0:
        _warn("Fig 2: no centralized curves found; saving empty figure")
    _save_pdf(fig, out_dir / "fig2_centralized_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 4 onwards (Fig 3 qualitative grid, Fig 4 ablation bars,
# Fig 5 federated curves, Fig 6 energy bars, tables, main) — TBD.
# ---------------------------------------------------------------------------


_FIG_REGISTRY = {
    # Populated incrementally as each block lands:
    2: ("fig2_centralized_curves", "fig2_centralized_curves.pdf"),
    # 3, 4, 5, 6 → wired in later blocks.
}


def main(argv=None) -> None:
    """Dispatch the requested figures + tables (each guarded by try/except
    so a single failing artefact does not abort the others)."""
    args = _parse_args(argv)
    fig_ids = _resolve_figs(args.figs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir={out_dir}  figs={fig_ids}  tables={args.tables}")

    # Currently-implemented dispatch.
    impls = {
        2: fig2_centralized_curves,
    }

    produced: List[str] = []
    skipped: List[str] = []
    for fid in fig_ids:
        fn = impls.get(fid)
        if fn is None:
            skipped.append(f"Fig{fid} (not yet implemented)")
            continue
        try:
            fn(args, out_dir)
            produced.append(_FIG_REGISTRY[fid][1])
        except Exception as e:                # noqa: BLE001
            _warn(f"Fig{fid} failed: {type(e).__name__}: {e}")
            skipped.append(f"Fig{fid} (error)")

    _log(f"produced: {produced or '(none)'}")
    if skipped:
        _log(f"skipped : {skipped}")


if __name__ == "__main__":
    main()
