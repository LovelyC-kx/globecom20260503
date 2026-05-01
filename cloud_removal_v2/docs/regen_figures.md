# Paper Figure Regeneration Cheat-Sheet

Run this end-to-end after the F\_snn $\alpha=0.1$ and $\alpha=0.01$
sweeps finish.  $\alpha=1.0$ is **excluded from the paper for now**;
the table row stays as `[XX.XX]` until v3.

All commands assume cwd is the project root
(`/root/autodl-tmp/shiyaunmingFLSNN-main/Decentralized-Satellite-FL-dev-main`).

---

## Step 0 — Pull the latest commits onto each container

```bash
git fetch origin claude/review-codebase-PZ315
git checkout claude/review-codebase-PZ315
git pull origin claude/review-codebase-PZ315
```

The two commits you need are:

* `4f27a10` — `fix(energy): persist pJ constants in JSON + freshen
  0.9→0.077 labels` (so `energy_summary.json` writes
  `ac_pj_per_op = 0.077` correctly).
* `d1817cd` — `fig(arch): fancy 3-D OrbitVLIF diagram replaces flat
  fig1.tex` (the new `plot_arch_diagram.py`).

---

## Step 1 — Re-run energy estimation (Fig 6 / 9 / 10 source)

The previous run wrote `ac_pj_per_op: null` because the CLI constants
weren't persisted into `summary["config"]`.  After pulling the fix
they will be written correctly.

```bash
python -m cloud_removal_v2.energy_estimation \
    --ckpt Outputs_v1/centralized_A1_vlif_cr1_best.pt \
    --data_root /root/autodl-tmp/C-CUHK \
    --source_subdir CUHK-CR1 \
    --split test \
    --num_samples 32 \
    --patch_size 64 \
    --backbone snn \
    --out_dir ./Outputs_energy_A1 \
    --ann_pj_per_mac 4.6 \
    --ac_pj_per_op  0.077 \
    --device cuda
```

Expected stdout (sanity check):

```
  E_ANN  (4.6 pJ/MAC)         : 20.75 mJ
  E_SNN_upper (4.6 pJ × r)    : 17.14 mJ   (1.21x)
  E_SNN_lower (0.077 pJ × r)  :  0.287 mJ  (72.30x)
```

---

## Step 2 — Render the new fancy architecture figure (fig1.pdf)

This **replaces** the old TikZ-rendered `fig1.tex` output.  The paper's
`\includegraphics{fig1.pdf}` path is unchanged.

```bash
python -m cloud_removal_v2.plot_arch_diagram \
    --out_dir ./figures \
    --out_name fig1.pdf
```

Three panels in one 8.5 x 5.4-in figure:

| Panel | Content                                                |
|-------|--------------------------------------------------------|
| (a)   | Single-sat U-Net: Input → PE → E1-3 → DSP → D3-1 → Head, with 3-D feature-map slabs and AGFM skip diamonds |
| (b)   | 5x10 Walker-Star: tilted ellipse rings + Gossip chain + dashed shortcut |
| (c)   | MFRB / 5QS / SHAM module zoom-ins                       |

---

## Step 3 — Regenerate every paper figure + LaTeX table in one shot

Adjust `--run_f_snn` if you launched the $\alpha = 0.1$ run with a
different `--run_name`.

```bash
python -m cloud_removal_v1.plot_paper_figs \
    --outputs_v1   ./Outputs_v1 \
    --outputs_v2   ./Outputs_v2 \
    --energy_dir   ./Outputs_energy_A1 \
    --out_dir      ./figures \
    --run_a1       A1_vlif_cr1 \
    --run_a2       A2_vlif_cr2 \
    --run_c2_cr1   C2_plain_ann_cr1 \
    --run_c2_cr2   C2_plain_ann_cr2 \
    --run_b1       B1_no_fsta_cr1 \
    --run_b2       B2_no_dual_group_cr1 \
    --run_b3       B3_binary_spike_cr1 \
    --run_f_snn    F_snn \
    --run_f_ann    F_ann \
    --run_f_plain  F_plain \
    --qual_dataset_root /root/autodl-tmp/C-CUHK \
    --device       cpu \
    --figs         all \
    --tables       yes
```

Outputs (every artefact is guarded by `try/except`, so a single missing
file doesn't abort the others):

| Fig | File                                  | Source data                                  |
|-----|---------------------------------------|----------------------------------------------|
| 2   | `fig2_centralized_curves.pdf`         | `Outputs_v1/centralized_A1/A2/C2_*.npz`     |
| 3   | `fig3_qualitative_grid.pdf`           | A1 ckpt + CUHK-CR test split                 |
| 4   | `fig4_ablation_bars.pdf`              | `Outputs_v1/centralized_B1/B2/B3_*.json`    |
| 5   | `fig5_federated_curves.pdf`           | `Outputs_v2/v2a_F_snn/F_ann/F_plain_*.npz` |
| 6   | `fig6_energy_bars.pdf`                | `Outputs_energy_A1/energy_summary.json`     |
| 7   | `fig7_centralized_4panel.pdf`         | A1/A2/C2 npz + summary                       |
| 9   | `fig9_per_layer_spike_rate.pdf`       | `Outputs_energy_A1/energy_summary.json`     |
| 10  | `fig10_per_layer_energy_paired.pdf`   | `Outputs_energy_A1/energy_summary.json`     |

Tables:

| Tab | File                          |
|-----|-------------------------------|
| I   | `figures/tab1_centralized_main.tex` |
| II  | `figures/tab2_ablation.tex`         |
| III | `figures/tab3_federated.tex`        |

---

## Step 4 — Fill Tab III $\alpha$-sensitivity rows by hand

`tab3_federated.tex` only renders the **one** locked F\_snn cell (the
$\alpha=0.1$ default).  The paper's Tab III also has $\alpha=0.01$ and
$\alpha=1.0$ rows that the auto-generator doesn't know about.  Extract
those numbers with:

```bash
python - <<'PY'
import json, glob
for run in ["F_snn", "F_snn_alpha001"]:        # add F_snn_alpha10 if you keep it
    matches = glob.glob(f"Outputs_v2/v2a_{run}_summary.json")
    if not matches:
        print(f"!! {run}: summary.json not found, skipping"); continue
    s = json.load(open(matches[0]))
    cell = s["final"].get("fedbn_Gossip_Averaging")
    if cell is None:
        print(f"!! {run}: cell fedbn_Gossip_Averaging missing"); continue
    psnr  = cell.get("PSNR_final")
    ssim  = cell.get("SSIM_final")
    comm  = cell.get("total_comm_bytes") / 1024**2
    wall  = cell.get("total_wall_seconds") / 3600
    print(f"{run:18s}  PSNR={psnr:.2f}  SSIM={ssim:.4f}  "
          f"Comm={comm:.0f} MB  Wall={wall:.2f} h")
PY
```

Then in `cloud_removal_v2/docs/paper_orbitvlif.tex` lines ~1015-1022,
replace the `[XX.XX]` placeholders.  F\_ann and F\_plain are already
fillable today (see below).

---

## Step 5 — F\_ann row (can do RIGHT NOW, no waiting)

F\_ann finished on 2026-04-30 23:26.  Pull the numbers and replace the
F\_ann line in Tab III:

```bash
python - <<'PY'
import json
s = json.load(open("Outputs_v2/v2a_F_ann_summary.json"))
c = s["final"]["fedbn_Gossip_Averaging"]
print(f"PSNR={c['PSNR_final']:.2f}  SSIM={c['SSIM_final']:.4f}  "
      f"Comm={c['total_comm_bytes']/1024**2:.0f} MB  "
      f"Wall={c['total_wall_seconds']/3600:.2f} h")
PY
```

---

## Notes / gotchas

* `--qual_dataset_root` is required for **Fig 3** only.  Skip it if
  you only want curves/tables.
* `--device cpu` is fine for Fig 3; the qualitative grid is 4-6
  inferences total, GPU is unnecessary.
* If `plot_paper_figs.py` warns `Fig N: ...skipped`, the corresponding
  npz / summary is missing — re-check the `--run_*` flags.
* Tab II caption hard-codes B1/B2/B3 names; if you renamed those runs,
  also pass `--run_b1 / --run_b2 / --run_b3` matching your `Outputs_v1/`.
* The paper currently builds with `pdflatex paper_orbitvlif.tex`; make
  sure `figures/` is on the include path (`\graphicspath{{./figures/}}`)
  or copy the pdfs next to the `.tex`.

---

## TL;DR — minimum command set after the runs finish

```bash
git pull
python -m cloud_removal_v2.energy_estimation --ckpt Outputs_v1/centralized_A1_vlif_cr1_best.pt --data_root /root/autodl-tmp/C-CUHK --source_subdir CUHK-CR1 --split test --num_samples 32 --patch_size 64 --backbone snn --out_dir ./Outputs_energy_A1 --ann_pj_per_mac 4.6 --ac_pj_per_op 0.077 --device cuda
python -m cloud_removal_v2.plot_arch_diagram --out_dir ./figures
python -m cloud_removal_v1.plot_paper_figs --outputs_v1 ./Outputs_v1 --outputs_v2 ./Outputs_v2 --energy_dir ./Outputs_energy_A1 --out_dir ./figures --qual_dataset_root /root/autodl-tmp/C-CUHK --figs all --tables yes
```

Then patch the four cells of Tab III by hand from the JSON dumps in
Steps 4 & 5.
