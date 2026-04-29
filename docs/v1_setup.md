# v1 Cloud-Removal Pipeline — Setup & Run Guide

This doc covers everything needed to get `run_v1_smoke.py` running on a
single NVIDIA V100 GPU.  For the higher-level plan (v1 vs v2 vs v3
scope) see the conversation log that accompanies this branch.

---

## 1. Environment

### Python + PyTorch
```bash
conda create -n flsnn_cr python=3.9 -y
conda activate flsnn_cr
# V100 is supported on either PyTorch 1.13 or 2.1; pick one.
conda install pytorch==2.1.* torchvision==0.16.* \
    pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### v1 pip deps
```bash
cd Decentralized-Satellite-FL-dev-main
pip install spikingjelly==0.0.0.0.14 thop tensorboard tqdm Pillow \
            scikit-image numpy matplotlib
# Optional (v1 default is backend='torch'): build cupy for your CUDA
# pip install cupy-cuda12x
```

If `spikingjelly` pulls a newer version incompatible with your torch,
pin `pip install torch==1.13.1` and rebuild spikingjelly from source
per its README.

---

## 2. Data

### CUHK-CR1 (primary; what the defaults assume)

**Source**: Sui et al., *"Diffusion Enhancement for Cloud Removal in
Ultra-Resolution Remote Sensing Imagery"*, TGRS 2024 — [arXiv 2401.15105](https://arxiv.org/abs/2401.15105).
**Download**: Baidu Pan link in the [original repo](https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal) (extraction code `bean`).

After download, arrange on disk as one of:

**Layout A** (recommended — script auto-detects train/test):
```
./data/CUHK-CR1/
    train/
        input/         # cloudy images  (.png / .jpg / .tif accepted)
        target/        # cloud-free matching pairs
    test/
        input/
        target/
```

**Layout B** (flat — script splits 8:2 randomly with `partition_seed`):
```
./data/CUHK-CR1/
    input/
    target/
```

Supported sub-folder names (either side): `input|cloudy|cloud` for
cloudy, `target|clear|label|gt` for clear.

### CUHK-CR2 (v2 scope)
Place the same way; `run_v1_smoke.py` does not read it in v1.

### RICE1 fallback
If the Baidu-Pan download is too slow, RICE1 works drop-in:
```bash
python run_v1_smoke.py --data_root ./data/RICE1 \
    --dataset_name RICE1 --run_name v1_rice_fallback
```

---

## 3. Run

### Smoke (defaults = agreed v1 config)
```bash
cd Decentralized-Satellite-FL-dev-main
python run_v1_smoke.py
```
This runs the **50/5/1 Walker Star** constellation (`num_planes=5,
sats_per_plane=10 → 50 satellites`), **10 global epochs**, **3
aggregation schemes** sequentially.  Estimated wall time on V100:
**3–5 hours**.

### Common overrides
```bash
# Shorter dry-run (3 epochs, smaller constellation) — ~20 min
python run_v1_smoke.py --num_epoch 3 --num_planes 3 --sats_per_plane 3 \
    --run_name dryrun

# Explicit data path + run tag
python run_v1_smoke.py --data_root /mnt/datasets/CUHK-CR1 \
    --run_name v1_cuhk_cr1

# If full-res eval OOMs (unlikely on V100 32GB, possible on 16GB)
python run_v1_smoke.py --eval_mode sliding

# Preview v2 FedBN behaviour early (still v1 schedule)
python run_v1_smoke.py --bn_local --run_name v1_with_fedbn
```

### Plots
```bash
python plot_v1_results.py --run_name v1_smoke
# → Outputs/v1_v1_smoke_{train_loss,test_psnr,test_ssim}.pdf
```

---

## 4. Outputs

```
Outputs/
    v1_smoke_<run>_Relaysum_Aggregation.npz
    v1_smoke_<run>_Gossip_Averaging.npz
    v1_smoke_<run>_All-Reduce_Aggregation.npz
    v1_smoke_<run>_summary.json          # final PSNR/SSIM + config snapshot
    v1_<run>_train_loss.pdf              # produced by plot_v1_results.py
    v1_<run>_test_psnr.pdf
    v1_<run>_test_ssim.pdf
tb/<run>/                                # tensorboard event files
```

Each `.npz` contains keys: `epochs, train_loss, eval_psnr, eval_ssim,
comm_bytes, wall_seconds, per_plane_psnr, per_plane_ssim`.

---

## 5. v1 success criteria

Borrowing the acceptance list from the v1 plan:
1. `run_v1_smoke.py` completes without exception.
2. All three schemes' **training loss monotonically decreases**.
3. Final PSNR ≥ **20 dB** (CUHK-CR1 is relatively easy).
4. Final SSIM ≥ **0.6**.
5. `RelaySum ≥ Gossip` (numerically; trend matches original FLSNN §VI-A).
6. Per-round bytes logged to tensorboard for later comm-vs-acc plot.

If any of these fail, do **not** move to v2.

---

## 6. Known caveats (v1)

* **No FedBN**: v1 aggregates BN + TDBN like every other layer
  (FedAvg baseline).  The `--bn_local` flag is already wired through
  `aggregation.py` — flip it on to get v2 semantics early.

* **IID partition**: v1 splits the training set uniformly across
  clients.  v2 adds Dirichlet-over-cluster partitioning.

* **RGB only**: CUHK-CR's NIR band is discarded in v1.  VLIFNet
  supports `inp_channels=4` via `build_vlifnet`, which v2 will use.

* **No ablations / baselines**: v1 produces the 3-scheme figure only.
  MultiSpike4-vs-NoisySpike, ANN-vs-SNN, and cloud-removal SOTA
  comparisons (ESDNet, SpA-GAN, centralized VLIFNet) are all v2.

* **Single-snapshot connectivity**: v1 uses a fixed chain topology
  (matches original FLSNN Fig 5 setup).  MDST routing and time-varying
  topology land in v3 alongside the 42/7/1 Walker Delta experiment.

---

## 7. If you hit a problem

1. `FileNotFoundError: Could not find paired input/target folders …`
   → recheck the on-disk layout (§2).

2. SpikingJelly import errors
   → `pip install --upgrade --force-reinstall spikingjelly==0.0.0.0.14`.

3. `CUDA out of memory` during eval
   → add `--eval_mode sliding`.

4. `CUDA out of memory` during training
   → drop `--train_batch_size 2` and/or `--patch_size 48`.

5. Numerical divergence in the first epoch
   → set `--lr 5e-4` (VLIFNet is moderately sensitive to LR at init).

6. Anything else — check the relevant risk entry in the v1 plan
   (R-v1-1 .. R-v1-8) for the documented remedy.
