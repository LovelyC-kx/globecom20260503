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
# Block 4 — Fig 3 qualitative grid (4 test images × 4 columns)
# ---------------------------------------------------------------------------

def _psnr_uint01(pred: np.ndarray, target: np.ndarray) -> float:
    """PSNR for two float32 [H, W, 3] arrays in [0, 1].  +inf when equal."""
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * float(np.log10(1.0)) - 10.0 * float(np.log10(mse))


def _score_cloud_severity(cloudy: np.ndarray, clear: np.ndarray) -> float:
    """Higher = harder image (more cloud coverage / larger cloudy-vs-GT
    deviation).  Use the L2 distance between normalised cloudy and clear
    arrays — a robust proxy that does not require explicit cloud masks.
    """
    return float(np.sqrt(np.mean((cloudy - clear) ** 2)))


def _select_test_indices(test_ds,
                         n: int = 4,
                         seed: int = 0,
                         topk_mult: int = 3) -> List[int]:
    """Deterministic + cloud-coverage-biased selection.

    Steps:
        1. Score every test pair by cloud severity (L2 cloudy-vs-GT).
        2. Take the top ``n * topk_mult`` indices (most cloudy).
        3. From that subset, sample ``n`` with a fixed-seed RNG so we
           still get a *deterministic* selection while not always picking
           the absolute top-N (avoids overcrowding visually-similar imgs).
    """
    if n <= 0:
        return []
    n_total = len(test_ds)
    if n_total == 0:
        return []
    if n_total <= n:
        return list(range(n_total))

    scores = np.empty(n_total, dtype=np.float64)
    for i in range(n_total):
        cloudy_t, clear_t = test_ds[i]
        cl  = cloudy_t.permute(1, 2, 0).cpu().numpy()
        gt  = clear_t.permute(1, 2, 0).cpu().numpy()
        scores[i] = _score_cloud_severity(cl, gt)

    topk = min(n_total, max(n, n * topk_mult))
    top_idx = np.argsort(-scores)[:topk]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(top_idx, size=n, replace=False)
    return sorted(chosen.tolist())


def _infer_image(model, x_chw: "torch.Tensor", device: "torch.device",
                 is_snn: bool) -> np.ndarray:
    """Run a single full-resolution image through ``model`` and return a
    [H, W, 3] float32 array in [0, 1].

    Mirrors v1's task.py contract: ``reset_net`` BEFORE and AFTER the
    forward pass when the backbone is spiking; no-op for ANN / Plain.
    """
    import torch
    model.eval()
    if is_snn:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
    with torch.no_grad():
        x = x_chw.unsqueeze(0).to(device)
        # Both PlainUNet and VLIFNet require H, W divisible by 4 — let the
        # model assertion fire if violated; we don't silently pad here.
        y = model(x)
    if is_snn:
        from spikingjelly.activation_based import functional
        functional.reset_net(model)
    y = y.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return y.astype(np.float32, copy=False)


def _resolve_qual_root(args, cfg: Optional[Dict]) -> Optional[str]:
    """Pick the test-set root for Fig 3 — explicit CLI flag first, else
    the data_root saved inside the A1 ckpt's config."""
    if args.qual_dataset_root:
        return args.qual_dataset_root
    if cfg is not None and isinstance(cfg.get("data_root"), str):
        return cfg["data_root"]
    return None


def fig3_qualitative_grid(args, out_dir: Path) -> None:
    """4 rows × 4 columns: Cloudy | VLIFNet | PlainUNet | Ground Truth.

    Each VLIFNet / PlainUNet cell shows a *prominent* PSNR label
    (top-left, white text on a dark filled rectangle) computed against
    the cell's ground truth.  Test images are selected deterministically
    via ``_select_test_indices`` (top-cloud-coverage bias + seed sample).

    Inputs (any of these missing → skip with warning, do not crash):
      * VLIFNet best ckpt   = Outputs_v1/centralized_<run_a1>_best.pt
      * PlainUNet best ckpt = Outputs_v1/centralized_<run_c2_cr1>_best.pt
      * Test dataset root   = --qual_dataset_root (else from A1 ckpt's
                              saved config['data_root'])

    Output: ``out_dir/fig3_qualitative_grid.pdf``.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt
    import torch

    outputs_v1 = Path(args.outputs_v1)
    device = torch.device(args.device)

    # 1) Load both ckpts.
    a1_ckpt = outputs_v1 / f"centralized_{args.run_a1}_best.pt"
    c2_ckpt = outputs_v1 / f"centralized_{args.run_c2_cr1}_best.pt"
    a1_loaded = load_ckpt_for_inference(a1_ckpt, device)
    c2_loaded = load_ckpt_for_inference(c2_ckpt, device)
    if a1_loaded is None and c2_loaded is None:
        _warn("Fig 3: both VLIFNet and PlainUNet ckpts missing; skipping figure")
        return
    a1_model, a1_cfg, a1_is_snn = a1_loaded if a1_loaded else (None, None, False)
    c2_model, c2_cfg, c2_is_snn = c2_loaded if c2_loaded else (None, None, False)

    # 2) Resolve test-set root (CLI override > A1 cfg > C2 cfg).
    root = _resolve_qual_root(args, a1_cfg) or _resolve_qual_root(args, c2_cfg)
    if root is None:
        _warn("Fig 3: no test dataset root resolvable (try --qual_dataset_root); "
              "skipping figure")
        return

    # 3) Build full-resolution test set (patch_size=None) and pick rows.
    try:
        from cloud_removal_v1.dataset import (
            PairedCloudDataset, derived_train_test_split,
        )
        try:
            test_ds = PairedCloudDataset(root, split="test", patch_size=None)
        except FileNotFoundError:
            _, test_sub = derived_train_test_split(
                root, patch_size_train=64, test_ratio=0.2, seed=args.qual_seed)
            test_ds = test_sub
    except Exception as e:                        # noqa: BLE001
        _warn(f"Fig 3: cannot open test set at {root}: "
              f"{type(e).__name__}: {e}; skipping figure")
        return

    n_rows = max(1, args.qual_n_samples)
    indices = _select_test_indices(test_ds, n=n_rows, seed=args.qual_seed)
    if not indices:
        _warn("Fig 3: empty test set; skipping figure")
        return

    # 4) Render grid.
    cols = ["Cloudy", "VLIFNet (ours)", "PlainUNet", "Ground Truth"]
    n_cols = len(cols)
    fig, axes = plt.subplots(
        len(indices), n_cols,
        figsize=(FIG_GRID[0], 1.85 * len(indices)),
        squeeze=False,
    )

    for r, idx in enumerate(indices):
        cloudy_t, clear_t = test_ds[idx]
        cloudy_img = cloudy_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)
        clear_img  = clear_t.permute(1, 2, 0).cpu().numpy().astype(np.float32)

        if a1_model is not None:
            try:
                vlif_img = _infer_image(a1_model, cloudy_t, device, a1_is_snn)
            except Exception as e:                # noqa: BLE001
                _warn(f"Fig 3 row {r}: VLIFNet inference failed "
                      f"({type(e).__name__}: {e}); blanking cell")
                vlif_img = None
        else:
            vlif_img = None

        if c2_model is not None:
            try:
                plain_img = _infer_image(c2_model, cloudy_t, device, c2_is_snn)
            except Exception as e:                # noqa: BLE001
                _warn(f"Fig 3 row {r}: PlainUNet inference failed "
                      f"({type(e).__name__}: {e}); blanking cell")
                plain_img = None
        else:
            plain_img = None

        cells = [
            (cloudy_img, None),
            (vlif_img,   _psnr_uint01(vlif_img, clear_img)
                         if vlif_img is not None else None),
            (plain_img,  _psnr_uint01(plain_img, clear_img)
                         if plain_img is not None else None),
            (clear_img,  None),
        ]

        for c, (img, psnr) in enumerate(cells):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#444444")
            if img is None:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center",
                        color=PALETTE_GRAY, fontsize=14)
            else:
                ax.imshow(np.clip(img, 0.0, 1.0), interpolation="nearest")
                if psnr is not None:
                    # Prominent: bold, top-left, white-on-black filled box.
                    ax.text(0.04, 0.92,
                            f"{psnr:.2f} dB",
                            transform=ax.transAxes,
                            ha="left", va="top",
                            fontsize=10, fontweight="bold", color="white",
                            bbox=dict(boxstyle="round,pad=0.18",
                                      facecolor="black",
                                      edgecolor="none",
                                      alpha=0.72))
            if r == 0:
                ax.set_title(cols[c], fontsize=10)

    fig.tight_layout(pad=0.25)
    _save_pdf(fig, out_dir / "fig3_qualitative_grid.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 5 — Fig 4 ablation bars (A1 vs B1 / B2 / B3, PSNR + SSIM)
# ---------------------------------------------------------------------------

def _smart_ylim(values: Sequence[float], pad: float
                ) -> Tuple[float, float]:
    """Auto Y-axis range so small ablation deltas are visible — pad below
    min and above max, never going below 0."""
    arr = np.array([v for v in values if v is not None and np.isfinite(v)],
                   dtype=float)
    if arr.size == 0:
        return (0.0, 1.0)
    lo = max(0.0, float(arr.min()) - pad)
    hi = float(arr.max()) + pad
    if hi - lo < 1e-6:
        hi = lo + max(pad, 1e-3)
    return (lo, hi)


def fig4_ablation_bars(args, out_dir: Path) -> None:
    """Two side-by-side subplots (PSNR | SSIM) comparing the full VLIFNet
    (A1) against three ablations:

        B1: −FSTA (FSTAModule + FreMLPBlock → Identity)
        B2: −DualGroup (PixelShuffle spatial path removed)
        B3: −MultiSpike4 (5-level → binary spikes)

    Bar heights = ``best.psnr`` / ``best.ssim`` from each run's summary.json.
    Top of each ablated bar carries a red ΔPSNR / ΔSSIM annotation
    (negative deltas in red, positive in green — for the rare case an
    ablation accidentally helps on noisy data).  A1's bar uses a ``//``
    hatch to flag the "reference / full model" condition.

    Skipped (with warning, no crash) if A1 summary is missing — there is
    no reference to compute deltas against.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt

    outputs_v1 = Path(args.outputs_v1)

    # Pull (run_name, label, color) — ORDER FIXED for paper consistency.
    spec = [
        (args.run_a1, "Full",         PALETTE_BLUE,   True),   # ← reference
        (args.run_b1, "−FSTA",        PALETTE_ORANGE, False),
        (args.run_b2, "−DualGroup",   PALETTE_GREEN,  False),
        (args.run_b3, "−MultiSpike4", PALETTE_GRAY,   False),
    ]

    # Load each summary; missing entries become None placeholders so the
    # bar order remains stable on partial data.
    rows: List[Tuple[str, str, Optional[float], Optional[float], bool]] = []
    for run_name, label, color, is_ref in spec:
        s = load_centralized_summary(outputs_v1, run_name)
        if s is None:
            rows.append((label, color, None, None, is_ref))
            continue
        best = s.get("best", {}) or {}
        psnr = best.get("psnr"); ssim = best.get("ssim")
        rows.append((label, color,
                     None if psnr is None or not np.isfinite(psnr) else float(psnr),
                     None if ssim is None or not np.isfinite(ssim) else float(ssim),
                     is_ref))

    a1_psnr = rows[0][2]; a1_ssim = rows[0][3]
    if a1_psnr is None and a1_ssim is None:
        _warn("Fig 4: A1 summary missing or empty; cannot compute deltas — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE_COL, sharey=False)

    metrics = [
        ("PSNR (dB)", [r[2] for r in rows], a1_psnr, 1.0,  "{:+.2f} dB"),
        ("SSIM",      [r[3] for r in rows], a1_ssim, 0.02, "{:+.4f}"),
    ]

    x = np.arange(len(rows))

    for ax, (ylabel, values, ref, pad, fmt) in zip(axes, metrics):
        ax.grid(True, axis="y", which="major")
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=15, ha="right")

        # Plot bars individually so each gets the right color + optional hatch.
        for i, ((label, color, psnr, ssim, is_ref), val) in enumerate(zip(rows, values)):
            if val is None:
                # Draw a faint placeholder bar so X-axis spacing stays correct.
                ax.bar(i, 0.0, color="none", edgecolor=PALETTE_GRAY,
                       linewidth=0.6, hatch="..")
                ax.text(i, 0.0, "no data", ha="center", va="bottom",
                        fontsize=7, color=PALETTE_GRAY, rotation=0)
                continue
            kwargs = dict(color=color, edgecolor="black", linewidth=0.6)
            if is_ref:
                kwargs["hatch"] = "//"
            ax.bar(i, val, **kwargs)

            # Delta annotation (red on regression, green on improvement).
            if not is_ref and ref is not None:
                delta = val - ref
                ann_color = "#C62828" if delta < 0 else "#2E7D32"
                ax.text(i, val, fmt.format(delta),
                        ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color=ann_color)

        ax.set_ylim(*_smart_ylim(values, pad=pad))

    axes[0].set_title("(a) PSNR")
    axes[1].set_title("(b) SSIM")

    fig.tight_layout(pad=0.4)
    _save_pdf(fig, out_dir / "fig4_ablation_bars.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 6 — Fig 5 federated convergence curves + cumulative-comm inset
# ---------------------------------------------------------------------------

def _per_plane_to_matrix(arr) -> Optional[np.ndarray]:
    """Convert run_smoke's object-dtype ``per_plane_psnr`` / ``per_plane_ssim``
    array (shape (n_epochs, n_planes), each cell a Python scalar) into a
    homogeneous float64 matrix.  Returns None if the structure is unusable.
    """
    if arr is None:
        return None
    try:
        rows: List[List[float]] = []
        for ep_row in np.atleast_1d(arr):
            # Each row is itself an iterable of per-plane scalars.
            vals = np.asarray(list(ep_row), dtype=np.float64).ravel()
            rows.append(vals.tolist())
    except Exception:                              # noqa: BLE001
        return None
    if not rows:
        return None
    n_planes = max(len(r) for r in rows)
    if n_planes == 0:
        return None
    out = np.full((len(rows), n_planes), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def fig5_federated_curves(args, out_dir: Path) -> None:
    """Single-panel PSNR-vs-round overlay for the three federated runs::

        F_SNN  (blue   'o')   — VLIFNet-SNN
        F_ANN  (orange 's')   — VLIFNet-ANN
        F_plain(green  'D')   — PlainUNet baseline

    Per-plane shading: ``fill_between(min, max)`` over per-plane PSNR
    arrays at α=0.18, drawn behind each mean curve.

    Inset (lower-right): cumulative communication volume in MB on a log
    Y-axis — strengthens the "VLIFNet does NOT cost more bandwidth than
    its ANN twin" claim referenced in §V.

    Skipped (no crash) only when ALL THREE runs are missing.
    """
    _setup_mpl()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    outputs_v2 = Path(args.outputs_v2)

    spec = [
        (args.run_f_snn,   "F-SNN (VLIFNet)",      PALETTE_BLUE,   "o"),
        (args.run_f_ann,   "F-ANN (VLIFNet ReLU)", PALETTE_ORANGE, "s"),
        (args.run_f_plain, "F-Plain (PlainUNet)",  PALETTE_GREEN,  "D"),
    ]

    drawn: List[Tuple[str, np.ndarray, np.ndarray]] = []   # (label, rounds, comm_cum_MB)
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    ax.grid(True, which="major")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("PSNR (dB)")

    legend_handles, legend_labels = [], []

    for run_name, label, color, marker in spec:
        d = load_federated_npz(outputs_v2, run_name,
                               bn_mode=args.f_bn, scheme=args.f_scheme)
        if d is None:
            continue
        rounds = d.get("epochs")
        if rounds is None or rounds.size == 0:
            _warn(f"Fig 5: {run_name} has no rounds; skipping curve")
            continue
        rounds = np.asarray(rounds, dtype=np.float64)

        # Prefer per-plane mean+envelope; fall back to flat eval_psnr if
        # per_plane_psnr is unusable (older runs / single-plane configs).
        per_plane = _per_plane_to_matrix(d.get("per_plane_psnr"))
        if per_plane is not None and per_plane.shape[0] == rounds.size:
            mean_psnr = np.nanmean(per_plane, axis=1)
            lo = np.nanmin(per_plane, axis=1)
            hi = np.nanmax(per_plane, axis=1)
            ax.fill_between(rounds, lo, hi, color=color, alpha=0.18,
                            linewidth=0.0)
        else:
            psnr_flat = d.get("eval_psnr")
            if psnr_flat is None:
                _warn(f"Fig 5: {run_name} missing eval_psnr; skipping curve")
                continue
            mean_psnr = np.asarray(psnr_flat, dtype=np.float64)

        mev = _smart_marker_every(rounds.size)
        (ln,) = ax.plot(rounds, mean_psnr,
                        color=color, marker=marker, markevery=mev,
                        label=label, linestyle="-")
        legend_handles.append(ln); legend_labels.append(label)

        comm = d.get("comm_bytes")
        if comm is not None and comm.size == rounds.size:
            comm_cum_MB = np.cumsum(np.asarray(comm, dtype=np.float64)) / (1024.0 ** 2)
            drawn.append((label, rounds, comm_cum_MB))
        else:
            drawn.append((label, rounds, None))

    if not legend_handles:
        _warn("Fig 5: no federated runs found; skipping figure")
        plt.close(fig)
        return

    ax.legend(legend_handles, legend_labels, loc="lower right",
              ncol=1)
    ax.margins(x=0.02)

    # tight_layout BEFORE adding the inset — inset_axes confuses
    # matplotlib's tight_layout solver and triggers a benign warning.
    fig.tight_layout(pad=0.3)

    # Inset — cumulative communication in MB, log scale.
    if any(c is not None for _, _, c in drawn):
        ax_in = inset_axes(ax, width="38%", height="32%",
                           loc="upper left",
                           bbox_to_anchor=(0.06, -0.04, 1.0, 1.0),
                           bbox_transform=ax.transAxes,
                           borderpad=0.0)
        for (label, rounds, comm_cum), (_, _, color, _) in zip(drawn, spec):
            if comm_cum is None:
                continue
            ax_in.plot(rounds, comm_cum, color=color, linewidth=1.0)
        ax_in.set_yscale("log")
        ax_in.set_xlabel("round", fontsize=6)
        ax_in.set_ylabel("MB (log)", fontsize=6)
        ax_in.tick_params(axis="both", which="both", labelsize=6, length=2)
        for spine in ax_in.spines.values():
            spine.set_linewidth(0.4)
        ax_in.grid(True, which="both", linewidth=0.3, alpha=0.4)

    _save_pdf(fig, out_dir / "fig5_federated_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Block 7 onwards (Fig 6 energy, tables, README polish) — TBD.
# ---------------------------------------------------------------------------


_FIG_REGISTRY = {
    # Populated incrementally as each block lands:
    2: ("fig2_centralized_curves", "fig2_centralized_curves.pdf"),
    3: ("fig3_qualitative_grid",   "fig3_qualitative_grid.pdf"),
    4: ("fig4_ablation_bars",      "fig4_ablation_bars.pdf"),
    5: ("fig5_federated_curves",   "fig5_federated_curves.pdf"),
    # 6 → wired in later block.
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
        3: fig3_qualitative_grid,
        4: fig4_ablation_bars,
        5: fig5_federated_curves,
    }

    produced: List[str] = []
    skipped: List[str] = []
    for fid in fig_ids:
        fn = impls.get(fid)
        if fn is None:
            skipped.append(f"Fig{fid} (not yet implemented)")
            continue
        out_name = _FIG_REGISTRY[fid][1]
        out_path = out_dir / out_name
        # Snapshot mtime so we can detect "function returned without
        # actually writing the file" (early-return on missing inputs).
        before_mtime = out_path.stat().st_mtime if out_path.exists() else None
        try:
            fn(args, out_dir)
        except Exception as e:                # noqa: BLE001
            _warn(f"Fig{fid} failed: {type(e).__name__}: {e}")
            skipped.append(f"Fig{fid} (error)")
            continue
        after_mtime = out_path.stat().st_mtime if out_path.exists() else None
        if after_mtime is not None and after_mtime != before_mtime:
            produced.append(out_name)
        else:
            skipped.append(f"Fig{fid} (skipped — missing inputs)")

    _log(f"produced: {produced or '(none)'}")
    if skipped:
        _log(f"skipped : {skipped}")


if __name__ == "__main__":
    main()
