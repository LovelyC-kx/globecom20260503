"""
Decentralized cloud-removal constellation orchestrator (v1).

Extends the training loop of `revised_satellite_system.ConstellationLearning`
for image regression.  We intentionally build a NEW class rather than
touching the original — the FLSNN classification pipeline stays reproducible.

Differences from the upstream class
-----------------------------------
1. One `CloudRemovalSNNTask` instance per (plane, satellite) — each owns
   its own VLIFNet, AdamW optimizer (persisted across rounds) and data
   loader.  No deepcopy of an nn.Module is ever performed.

2. Aggregation routines live in aggregation.py (C6).  This module only
   orchestrates who-talks-to-whom and when.

3. Communication cost (bytes transmitted per round per ISL) is logged
   to feed the v2 "accuracy vs communication" plot.

4. All three baseline inter-plane schemes are supported:
       ALLREDUCE : every plane averages every plane's intra-plane model.
       GOSSIP    : each plane only averages with immediate chain neighbours.
       RELAYSUM  : RelaySum (Vogels et al., NeurIPS 2021).

5. Intra-plane aggregation = equal-weight average (ring-all-reduce's
   mathematical equivalent; avoids implementing the actual ring stages
   when we're single-process simulated).

Not included in v1 (hooks left for later versions)
-------------------------------------------------
* FedBN / BN-local aggregation toggle — already wired via
  `CloudRemovalSNNTask.apply_global_weights(bn_local=...)` (C6 will flip
  the default when v2 ablations request it).
* HarmoFL amplitude sharing (v3).
* Energy-aware MDST edge weights (v3).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import GOSSIP, RELAYSUM, ALLREDUCE
from cloud_removal_task import CloudRemovalSNNTask
from aggregation import (
    is_bn_key,
    average_state_dicts,
    state_div,
    zeros_like_state,
)


def _is_tensor(x) -> bool:
    return isinstance(x, torch.Tensor)


# ---------------------------------------------------------------------------
# Byte-counting helper (for the v2 comm-vs-acc figure)
# ---------------------------------------------------------------------------

def _state_dict_bytes(sd: Dict[str, torch.Tensor]) -> int:
    """Rough on-wire byte count of a fp32 state-dict.  Non-tensor entries
    (e.g. SpikingJelly's un-initialised neuron memory scalars) contribute 0."""
    total = 0
    for t in sd.values():
        if _is_tensor(t):
            total += t.numel() * t.element_size()
    return total


def _zero_state_on_device(reference_sd: Dict[str, torch.Tensor],
                          device: torch.device) -> Dict[str, torch.Tensor]:
    """zeros_like for every tensor value, placed on `device`.  Non-tensor
    values (float scalars, None) are preserved verbatim.  Used to seed
    RelaySum relay buffers on the GPU so they can be aggregated alongside
    GPU-resident intra-plane weights without a device-mismatch error."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in reference_sd.items():
        if _is_tensor(v):
            out[k] = torch.zeros_like(v, device=device)
        else:
            out[k] = v
    return out


# Aggregation primitives come from aggregation.py (C6).
# Everything in this file that mutates a state-dict goes through
# average_state_dicts / state_div so that the bn_local knob is honoured
# from a single place.


# ---------------------------------------------------------------------------
# Constellation
# ---------------------------------------------------------------------------


class CloudRemovalConstellation:
    """Top-level orchestrator.

    Parameters
    ----------
    num_planes : int
    sats_per_plane : int
    client_datasets : list[list[Dataset]]
        Shape = (num_planes, sats_per_plane); each inner entry is the
        local dataset subset for that satellite.
    args : Namespace
        v1 expects: lr, wd, num_workers, train_batch_size, local_iters,
        intra_plane_iters, T, vlif_dim, en_blocks, de_blocks,
        num_epoch, clip_grad, ssim_weight, charbonnier_eps,
        vlif_backend, min_lr, warmup_epochs, bn_local (bool, v1=False).
    init_state_dict : dict | None
        Shared initial weights for every satellite.  If None, the first
        satellite's fresh init is broadcast to all others.
    device : str
    logger : callable | None
        `logger(msg: str)` — defaults to print.
    """

    def __init__(self,
                 num_planes: int,
                 sats_per_plane: int,
                 client_datasets: Sequence[Sequence[Dataset]],
                 args,
                 init_state_dict: Optional[Dict[str, torch.Tensor]] = None,
                 device: str = "cuda",
                 logger: Optional[Callable[[str], None]] = None,
                 ):
        assert len(client_datasets) == num_planes
        for row in client_datasets:
            assert len(row) == sats_per_plane

        self.num_planes = num_planes
        self.sats_per_plane = sats_per_plane
        self.args = args
        self.device = torch.device(device)
        self.log = logger if logger is not None else (lambda m: print(m, flush=True))

        # ---------------- Per-satellite tasks --------------------------
        # Build the first satellite first to get a concrete state_dict to
        # seed the rest.
        self.log(f"[init] building {num_planes}×{sats_per_plane} "
                 f"= {num_planes * sats_per_plane} satellite tasks")
        self.planes: List[List[CloudRemovalSNNTask]] = []
        seed_sd = init_state_dict
        for p in range(num_planes):
            row: List[CloudRemovalSNNTask] = []
            for s in range(sats_per_plane):
                task = CloudRemovalSNNTask(
                    args=args,
                    local_dataset=client_datasets[p][s],
                    init_state_dict=seed_sd,
                    device=str(self.device),
                )
                if seed_sd is None:
                    seed_sd = task.get_weights(cpu=True)
                row.append(task)
            self.planes.append(row)
        self.init_state_dict = seed_sd
        self.log("[init] satellite tasks ready")

        # ---------------- RelaySum state (per-plane) -------------------
        # Shapes match upstream revised_satellite_system.ConstellationLearning.
        # IMPORTANT: the relay buffers must live on the SAME device as the
        # per-satellite VLIFNet weights, otherwise RelaySum's `add_`/`copy_`
        # operations raise "Expected all tensors to be on the same device".
        # `seed_sd` is a CPU snapshot coming from get_weights(cpu=True); we
        # zero it on the constellation's device before deepcopy'ing into
        # the relay lattice.
        self._zero = _zero_state_on_device(seed_sd, self.device)
        self.received_relay_weights: List[List[Dict[str, torch.Tensor]]] = []
        self.transmitted_relay_weights: List[List[Dict[str, torch.Tensor]]] = []
        self.received_relay_counts: List[List[int]] = []
        self.transmitted_relay_counts: List[List[int]] = []
        self._reset_relay_buffers()

        # ---------------- Connectivity (inter-plane) -------------------
        # v1: fixed chain by default; caller may override via
        # `set_connectivity_matrix(...)` before first round.
        self.connectivity_matrix: np.ndarray = self._chain_topology(num_planes)

        # ---------------- Metrics bookkeeping --------------------------
        self.round_bytes: List[int] = []
        self.round_train_loss: List[float] = []

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chain_topology(n: int) -> np.ndarray:
        m = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            m[i, i] = 1
            if i > 0:
                m[i, i - 1] = 1
            if i < n - 1:
                m[i, i + 1] = 1
        return m

    def set_connectivity_matrix(self, mat: np.ndarray) -> None:
        assert mat.shape == (self.num_planes, self.num_planes)
        self.connectivity_matrix = mat.astype(np.int32).copy()

    def _reset_relay_buffers(self) -> None:
        self.received_relay_weights = [
            [deepcopy(self._zero) for _ in range(self.num_planes)]
            for _ in range(self.num_planes)
        ]
        self.transmitted_relay_weights = [
            [deepcopy(self._zero) for _ in range(self.num_planes)]
            for _ in range(self.num_planes)
        ]
        self.received_relay_counts = [
            [0 for _ in range(self.num_planes)] for _ in range(self.num_planes)
        ]
        self.transmitted_relay_counts = [
            [0 for _ in range(self.num_planes)] for _ in range(self.num_planes)
        ]

    def reset_all(self, new_init: Optional[Dict[str, torch.Tensor]] = None
                  ) -> None:
        """Reset training state for a fresh run (used when switching
        between RelaySum / Gossip / AllReduce in the same script)."""
        sd = new_init if new_init is not None else self.init_state_dict
        for row in self.planes:
            for task in row:
                task.apply_global_weights(sd, bn_local=False)
                task.global_round = 0
                # Rebuild optimizer to purge moment buffers from the prior run
                task.optimizer = torch.optim.AdamW(
                    task.model.parameters(),
                    lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=getattr(self.args, "wd", 0.0),
                )
        self._reset_relay_buffers()
        self.round_bytes = []
        self.round_train_loss = []

    # ------------------------------------------------------------------
    # One full global round
    # ------------------------------------------------------------------

    def train_one_round(self, aggregation_scheme: str) -> float:
        """Execute (intra_plane_iters × local) local training + one
        inter-plane aggregation step.  Returns the round's mean train loss."""
        assert aggregation_scheme in (GOSSIP, RELAYSUM, ALLREDUCE)

        bn_local = bool(getattr(self.args, "bn_local", False))
        total_rounds = self.args.num_epoch
        warmup = getattr(self.args, "warmup_epochs", 3)

        # ---- 1. Intra-plane rounds (local train + intra avg) ----------
        round_losses: List[float] = []
        for plane_iter in range(self.args.intra_plane_iters):
            # Local training for every satellite
            for p in range(self.num_planes):
                for s in range(self.sats_per_plane):
                    task = self.planes[p][s]
                    loss, _, _ = task.local_training(
                        total_global_rounds=total_rounds,
                        warmup_rounds=warmup,
                    )
                    round_losses.append(loss)
            # Intra-plane averaging — equal-weight over that plane's sats.
            # bn_local=False on the intra-plane step even when FedBN is on:
            # satellites in the SAME plane share a data distribution, so
            # averaging their BN stats is still beneficial.  Only the
            # INTER-plane step (below) respects bn_local.
            for p in range(self.num_planes):
                plane_avg = average_state_dicts(
                    [t.get_weights(cpu=False) for t in self.planes[p]],
                    bn_local=False,
                )
                for t in self.planes[p]:
                    t.apply_global_weights(plane_avg, bn_local=False)

        # ---- 2. Inter-plane aggregation -------------------------------
        intra_plane_weights = [
            self.planes[p][0].get_weights(cpu=False)    # all sats in plane p are identical after intra avg
            for p in range(self.num_planes)
        ]

        bytes_this_round = 0

        if aggregation_scheme == ALLREDUCE:
            allreduced = average_state_dicts(intra_plane_weights, bn_local=bn_local)
            new_plane_states = [allreduced for _ in range(self.num_planes)]
            # In all-reduce the total bytes per round = N * |state_dict|
            bytes_this_round = self.num_planes * _state_dict_bytes(allreduced)

        elif aggregation_scheme == GOSSIP:
            new_plane_states = self._gossip_average(intra_plane_weights, bn_local=bn_local)
            # Gossip: each plane sends to up to 2 neighbours per round.
            bytes_this_round = 0
            for p in range(self.num_planes):
                neighbours = int(self.connectivity_matrix[p].sum() - 1)  # excl self
                bytes_this_round += neighbours * _state_dict_bytes(intra_plane_weights[p])

        else:   # RELAYSUM
            new_plane_states = self._relaysum_step(intra_plane_weights, bn_local=bn_local)
            # RelaySum: each plane sends one relay + counter to each neighbour.
            bytes_this_round = 0
            for p in range(self.num_planes):
                neighbours = int(self.connectivity_matrix[p].sum() - 1)
                bytes_this_round += neighbours * _state_dict_bytes(intra_plane_weights[p])

        # ---- 3. Broadcast new plane state to each plane's satellites ---
        for p in range(self.num_planes):
            for t in self.planes[p]:
                t.apply_global_weights(new_plane_states[p], bn_local=bn_local)
                t.cleanup_between_rounds()

        mean_loss = float(np.mean(round_losses)) if round_losses else float("nan")
        self.round_train_loss.append(mean_loss)
        self.round_bytes.append(bytes_this_round)
        return mean_loss

    # ------------------------------------------------------------------
    # Inter-plane aggregation primitives
    # ------------------------------------------------------------------

    def _gossip_average(self,
                        intra_plane_weights: Sequence[Dict[str, torch.Tensor]],
                        bn_local: bool = False,
                        ) -> List[Dict[str, torch.Tensor]]:
        """Chain-gossip: plane p averages its weights with its neighbours
        on the connectivity graph."""
        n = self.num_planes
        new_states: List[Dict[str, torch.Tensor]] = []
        for p in range(n):
            # Equal-weight over (p itself ∪ its neighbours)
            neigh = [q for q in range(n)
                     if self.connectivity_matrix[p, q] == 1 and q != p]
            participants = [intra_plane_weights[p]] + [intra_plane_weights[q] for q in neigh]
            new_states.append(average_state_dicts(participants, bn_local=bn_local))
        return new_states

    def _relaysum_step(self,
                       intra_plane_weights: Sequence[Dict[str, torch.Tensor]],
                       bn_local: bool = False,
                       ) -> List[Dict[str, torch.Tensor]]:
        """One RelaySum round — follows Algorithm 2 of the FLSNN paper and
        mirrors revised_satellite_system.ConstellationLearning.  Operates on
        the persistent per-plane relay buffers.

        When bn_local=True, BN-tagged keys are kept local to each plane by
        (a) leaving them out of the relay buffers (so they never travel), and
        (b) overwriting them back with the plane's own pre-aggregation values
            at the end.  This guarantees FedBN semantics even when RelaySum's
            delayed-delivery buffers are in flight.
        """
        n = self.num_planes

        def _clone(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Deep-clone tensor entries; pass non-tensor entries through."""
            return {k: (v.detach().clone() if _is_tensor(v) else v)
                    for k, v in sd.items()}

        # Snapshot each plane's BN-tagged values BEFORE aggregation so we
        # can restore them verbatim if bn_local is set.
        bn_snapshots: Optional[List[Dict[str, torch.Tensor]]] = None
        if bn_local:
            bn_snapshots = []
            for p in range(n):
                bn_snapshots.append({
                    k: v.detach().clone()
                    for k, v in intra_plane_weights[p].items()
                    if is_bn_key(k) and _is_tensor(v)
                })

        # --- Build the tailored messages that plane p will send to each q ---
        for p in range(n):
            for q in range(n):
                if self.connectivity_matrix[p, q] == 1 and p != q:
                    new_relay = _clone(intra_plane_weights[p])
                    new_count = 1
                    for l in range(n):
                        if self.connectivity_matrix[l, p] == 1 and l != q and l != p:
                            for k in new_relay:
                                a = new_relay[k]
                                b = self.received_relay_weights[p][l].get(k)
                                if _is_tensor(a) and _is_tensor(b):
                                    a.add_(b)
                            new_count += self.received_relay_counts[p][l]
                    self.transmitted_relay_weights[p][q] = new_relay
                    self.transmitted_relay_counts[p][q] = new_count

        # --- Each plane receives + aggregates ---
        new_states: List[Dict[str, torch.Tensor]] = []
        for p in range(n):
            agg = _clone(intra_plane_weights[p])
            count = 1
            for q in range(n):
                if self.connectivity_matrix[q, p] == 1 and q != p:
                    # "Receive" : copy transmitted buffer into received slot
                    self.received_relay_weights[p][q] = self.transmitted_relay_weights[q][p]
                    self.received_relay_counts[p][q]  = self.transmitted_relay_counts[q][p]
                    for k in agg:
                        a = agg[k]
                        b = self.received_relay_weights[p][q].get(k)
                        if _is_tensor(a) and _is_tensor(b):
                            a.add_(b)
                    count += self.received_relay_counts[p][q]
            state_div(agg, count)

            # FedBN restoration: replace aggregated BN values with the
            # local pre-aggregation snapshot.  This is safe because the
            # relay buffers themselves will *eventually* also propagate
            # BN keys, but by the time they arrive they are applied to a
            # plane that has already over-written them locally.
            if bn_snapshots is not None:
                for k, v in bn_snapshots[p].items():
                    dst = agg.get(k)
                    if _is_tensor(dst) and _is_tensor(v):
                        dst.copy_(v)

            new_states.append(agg)

        return new_states

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot_plane_weights(self) -> List[Dict[str, torch.Tensor]]:
        """CPU copy of current per-plane model weights (for external eval)."""
        return [self.planes[p][0].get_weights(cpu=True) for p in range(self.num_planes)]
