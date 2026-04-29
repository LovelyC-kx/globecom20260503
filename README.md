# Decentralized-Satellite-FL-dev

Decentralized federated learning for satellites, evolved from the
FLSNN classification paper (arXiv:2501.15995) into a cloud-removal
regression pipeline.

## Tracks

This repository hosts two parallel pipelines:

1. **Legacy classification** (unchanged)
   Original FLSNN framework for land-cover classification on EuroSAT.
   Entry points: `main.py`, `parallel_training.py`,
   `aggregation_comparison.py`, `ann_snn_comparison.py`,
   `aggregation_tree_comparison.py`.

2. **v1 cloud removal** (new)
   Decentralized training of VLIFNet for optical cloud removal on
   CUHK-CR1 / RICE1.  Entry point: `run_v1_smoke.py`.
   See [docs/v1_setup.md](docs/v1_setup.md) for the complete guide.

The two tracks share the original `revised_satellite_system.py`,
`utils.py`, and `STK_simulator/` modules — v1 adds a parallel set of
files under the `cloud_removal_*` prefix and does not modify any
existing behaviour.

## v1 quickstart

```bash
# 1. Env
conda create -n flsnn_cr python=3.9 -y && conda activate flsnn_cr
conda install pytorch==2.1.* torchvision==0.16.* pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
cd Decentralized-Satellite-FL-dev-main
pip install spikingjelly==0.0.0.0.14 thop tensorboard tqdm Pillow \
            scikit-image numpy matplotlib

# 2. Data — place CUHK-CR1 at ./data/CUHK-CR1/{train,test}/{input,target}/

# 3. Smoke run (3 schemes × 10 epochs on a 50-satellite constellation)
python run_v1_smoke.py

# 4. Plots
python plot_v1_results.py --run_name v1_smoke
```

## v1 file map

| Role | File |
|------|------|
| Model (VLIFNet + FSTA) | `Spiking_Models/VLIFNet/{model.py, fsta_module.py}` |
| Paired cloud dataset + partitioner | `cloud_removal_dataset.py` |
| Per-satellite trainer (Charbonnier + SSIM loss, reset_net safe) | `cloud_removal_task.py` |
| Decentralized orchestrator (RelaySum / Gossip / AllReduce) | `cloud_removal_constellation.py` |
| BN-key-aware aggregation primitives (v2 FedBN hook) | `aggregation.py` |
| PSNR / SSIM evaluator (full-image + sliding-window) | `cloud_removal_eval.py` |
| v1 hyperparameter defaults + CLI | `cloud_removal_config.py` |
| Entry point for v1 experiments | `run_v1_smoke.py` |
| Plotting | `plot_v1_results.py` |
| Docs | `docs/v1_setup.md` |

## Design principles for v1 → v3 evolution

* Every v1 file uses `cloud_removal_` prefix to stay out of the
  classification pipeline's namespace.
* Five extension flags are already wired through (default off in v1):
    * `bn_local=False` in `aggregation.average_state_dicts` → v2 flips
      for FedBN.
    * `spike_fn_cls=MultiSpike4` in `Spiking_Models/VLIFNet/model.py`
      (v2 NoisySpike ablation).
    * `preprocessor=None` hook in `CloudRemovalSNNTask` (v3 HarmoFL).
    * `mode='basic'` in the edge-weight computation path (v3 TE-MDST).
    * `aux_loss_weight=0.0` in the task (v3 auxiliary supervision).
* RelaySum's per-plane persistent buffers are preserved across
  aggregation-scheme switches inside a single run; `reset_all()` is
  idempotent.
