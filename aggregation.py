"""
Federated aggregation primitives (v1 ↔ v3 shared).

Scope of this module
--------------------
Pure tensor-level operations over a list of state-dicts.  No knowledge
of the satellite / plane abstraction — that lives in
cloud_removal_constellation.py.

Why a separate file?
  * v1 : keep aggregation logic unit-testable in isolation.
  * v2 : FedBN just flips `bn_local=True` below.
  * v3 : HarmoFL amplitude sharing will add `amp_share=True`, and
         TE-MDST will swap out the constellation-level edge-weight
         policy but keep these primitives untouched.

The flags are additive: any unrecognised kwarg raises TypeError (rather
than silently being ignored), which makes later refactors fail loud.

Public surface
--------------
* is_bn_key(key)            -> bool
* average_state_dicts(...)  -> Dict[str, Tensor]
* apply_aggregated(...)     -> None  (in-place update of a task's sd)
* State math helpers        : state_add / state_mul / state_div
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Key classification — the single source of truth for "is this a BN-ish
# parameter?".  cloud_removal_task.is_bn_key imports this.
# ---------------------------------------------------------------------------

# Order matters: longer / more specific fragments come first so that
# `running_mean` is matched via `running_` (BN-tagged) before the generic
# `mean` substring could ever fire.
_BN_FRAGMENTS = (
    "running_",               # torch.nn.BatchNorm*
    "num_batches_tracked",
    "threshold",              # spikingjelly TDBN flag
    ".bn",                    # nn.BatchNorm2d named `bn`, `bn1`, ...
    "norm.",                  # LayerNorm, GroupNorm, InstanceNorm
    "_bn.",
    "bns.",                   # spikingjelly TemporalBN (`bns.0.weight` etc.)
)


def is_bn_key(key: str) -> bool:
    """Heuristic: does this state_dict key belong to a BN / normaliser?

    Designed to catch:
      * torch.nn.BatchNorm{1d,2d,3d}.{weight,bias,running_mean,running_var,
                                      num_batches_tracked}
      * spikingjelly ThresholdDependentBatchNorm2d (anything under .threshold*)
      * spikingjelly TemporalBN (bns.0.*, bns.1.*, ...)
      * GroupNorm / LayerNorm / InstanceNorm under a `norm` or `_bn` attribute

    and NOT catch weight/bias of ordinary Conv2d / Linear named `conv`,
    `fc`, `proj`, ... — their keys don't contain any of the fragments above.
    """
    kl = key.lower()
    return any(fragment in kl for fragment in _BN_FRAGMENTS)


# ---------------------------------------------------------------------------
# In-place state-dict math
# ---------------------------------------------------------------------------

def _is_tensor(x) -> bool:
    """True iff x is a torch.Tensor we can aggregate.

    SpikingJelly MemoryModule._save_to_state_dict() writes scalar floats
    (e.g. `v = 0.0`) or `None` for neuron membrane potentials that have
    never been instantiated.  These must not participate in aggregation:
      - they are ephemeral per-forward state, not learnable parameters;
      - torch.zeros_like(0.0) / (0.0).detach() raise AttributeError.
    """
    return isinstance(x, torch.Tensor)


def state_add(acc: Dict[str, torch.Tensor],
              other: Dict[str, torch.Tensor],
              *, scale: float = 1.0) -> None:
    """acc ← acc + scale · other (per-key, in place).  Non-tensor keys skipped."""
    for k, v in other.items():
        if _is_tensor(v) and _is_tensor(acc.get(k)):
            acc[k].add_(v, alpha=scale)


def state_mul(sd: Dict[str, torch.Tensor], scale: float) -> None:
    """sd ← scale · sd (in place).  Non-tensor keys skipped."""
    for v in sd.values():
        if _is_tensor(v):
            v.mul_(scale)


def state_div(sd: Dict[str, torch.Tensor], scale: float) -> None:
    """sd ← sd / scale (in place).  Non-tensor keys skipped."""
    assert scale != 0
    inv = 1.0 / float(scale)
    for v in sd.values():
        if _is_tensor(v):
            v.mul_(inv)


def zeros_like_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """zeros_like for every tensor value; non-tensor values copied as-is.

    IMPORTANT: unlike `torch.zeros_like`, this preserves the original
    non-tensor scalars (e.g. SpikingJelly's un-initialised `v = 0.0`),
    which lets a down-stream `.load_state_dict(...)` still succeed.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        out[k] = torch.zeros_like(v) if _is_tensor(v) else v
    return out


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------

def average_state_dicts(
    states: Sequence[Dict[str, torch.Tensor]],
    weights: Optional[Sequence[float]] = None,
    *,
    bn_local: bool = False,                  # v2 knob
    amp_share: bool = False,                 # v3 knob (placeholder)
) -> Dict[str, torch.Tensor]:
    """Weighted average of a list of state-dicts.

    Parameters
    ----------
    states : list of dicts
        All must share the same keys and per-key shapes.
    weights : list of floats | None
        If None, uniform average is used.  Must sum to > 0.
    bn_local : bool
        When True, BN-tagged keys (see is_bn_key) are NOT averaged — the
        output keeps the values from `states[0]` unchanged.  Callers are
        expected to THEN load this aggregated dict via
        apply_aggregated(..., bn_local=True) so each client keeps its own
        BN state.  Implementing the skip here AND in apply_aggregated is
        deliberate belt-and-braces: whatever the caller forgets, BN stays
        local.
    amp_share : bool
        v3 placeholder for HarmoFL amplitude averaging on spectral keys.
        v1 and v2 must pass amp_share=False.

    Returns
    -------
    Aggregated state-dict (fresh tensors, same device as states[0]).
    """
    if amp_share:
        raise NotImplementedError(
            "amp_share is scheduled for v3 (HarmoFL).  v1/v2 must use amp_share=False.")

    assert len(states) > 0, "average_state_dicts: empty input"
    ref_keys = list(states[0].keys())
    for i, sd in enumerate(states[1:], start=1):
        if list(sd.keys()) != ref_keys:
            raise KeyError(
                f"state_dict[{i}] keys differ from state_dict[0]; "
                f"aggregation would silently drop parameters.")

    if weights is None:
        weights = [1.0] * len(states)
    assert len(weights) == len(states)
    total_w = float(sum(weights))
    assert total_w > 0, "average_state_dicts: weights sum to 0"

    out: Dict[str, torch.Tensor] = {}
    for k in ref_keys:
        v0 = states[0][k]
        # Preserve non-tensor values (e.g. SpikingJelly un-initialised
        # neuron memory `v = 0.0` scalars) and BN-tagged tensors verbatim.
        if not _is_tensor(v0) or (bn_local and is_bn_key(k)):
            out[k] = v0.detach().clone() if _is_tensor(v0) else v0
            continue
        acc = v0.detach().clone() * float(weights[0])
        for i in range(1, len(states)):
            acc.add_(states[i][k], alpha=float(weights[i]))
        acc.div_(total_w)
        out[k] = acc
    return out


# ---------------------------------------------------------------------------
# In-place application to a task
# ---------------------------------------------------------------------------

def apply_aggregated(
    target_sd: Dict[str, torch.Tensor],
    global_sd: Dict[str, torch.Tensor],
    *,
    bn_local: bool = False,
) -> None:
    """Copy `global_sd` into `target_sd` in place, with optional BN skip.

    target_sd is typically `model.state_dict()` on the satellite.  Note
    that mutating this dict's tensors in place is sufficient — they are
    aliased with the model's parameters.

    Non-tensor entries (e.g. SpikingJelly un-initialised neuron memory
    `v = 0.0` or `None`) are left untouched on the target; they will be
    re-initialised by the next `functional.reset_net(...)` call anyway.
    """
    for k, v in global_sd.items():
        if bn_local and is_bn_key(k):
            continue
        if k not in target_sd:
            continue
        dst = target_sd[k]
        if not _is_tensor(v) or not _is_tensor(dst):
            continue
        dst.copy_(v.to(dst.device, dtype=dst.dtype, non_blocking=True))


# ---------------------------------------------------------------------------
# Sanity check — run this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # ---------------- Test 1: is_bn_key classification ----------------
    samples = [
        # Should be True
        ("encoder_level1.initial_residual.bn1.threshold",        True),
        ("encoder_level1.initial_residual.bn1.weight",           True),
        ("encoder_level1.initial_residual.bn1.bias",             True),
        ("encoder_level1.initial_residual.bn1.running_mean",     True),
        ("encoder_level1.initial_residual.bn1.running_var",      True),
        ("encoder_level1.initial_residual.bn1.num_batches_tracked", True),
        ("fre_mlp.norm.weight",                                  True),
        ("fre_mlp.norm.bias",                                    True),
        # Should be False
        ("patch_embed.proj.weight",                              False),
        ("encoder_level1.initial_residual.conv1.weight",         False),
        ("encoder_level1.initial_residual.high_freq_scale_1",    False),
        ("encoder_level1.initial_residual.cross_scale_gate",     False),
        ("output.0.weight",                                      False),
        ("aux_head_level2.weight",                               False),
    ]
    bad = 0
    for key, expected in samples:
        got = is_bn_key(key)
        flag = "OK " if got == expected else "!! "
        if got != expected:
            bad += 1
        print(f"{flag}{key:60s}  expect={expected}  got={got}")
    if bad:
        print(f"FAILED: {bad} mis-classified keys", file=sys.stderr)
        sys.exit(1)
    print("is_bn_key: all 14 sample keys classified correctly")

    # ---------------- Test 2: average_state_dicts tolerates non-tensor entries ----
    # Simulate SpikingJelly's mixed state_dict: float scalars + None + tensors.
    sd_a = {
        "conv.weight":                     torch.ones(2, 2),
        "lif.v":                           0.0,        # SpikingJelly un-init memory
        "lif.spike":                       None,       # another SJ sentinel
        "bn.weight":                       torch.tensor([1.0, 1.0]),
        "bn.running_mean":                 torch.tensor([0.5, 0.5]),
    }
    sd_b = {k: (v.clone() * 2.0 if _is_tensor(v) else v) for k, v in sd_a.items()}
    out = average_state_dicts([sd_a, sd_b])
    assert torch.allclose(out["conv.weight"], torch.full((2, 2), 1.5))
    assert out["lif.v"] == 0.0
    assert out["lif.spike"] is None
    assert torch.allclose(out["bn.weight"], torch.full((2,), 1.5))
    print("average_state_dicts: non-tensor entries (float scalar, None) preserved")

    # ---------------- Test 3: apply_aggregated tolerates non-tensor entries ------
    tgt = {
        "conv.weight":     torch.zeros(2, 2),
        "lif.v":           0.0,
        "lif.spike":       None,
    }
    src = {
        "conv.weight":     torch.full((2, 2), 9.0),
        "lif.v":           42.0,   # should be ignored
        "lif.spike":       None,
    }
    apply_aggregated(tgt, src)
    assert torch.allclose(tgt["conv.weight"], torch.full((2, 2), 9.0))
    assert tgt["lif.v"] == 0.0, f"expected lif.v unchanged (0.0), got {tgt['lif.v']}"
    print("apply_aggregated: non-tensor entries left untouched")

    # ---------------- Test 4: zeros_like_state preserves scalars ----------------
    zeros = zeros_like_state(sd_a)
    assert torch.all(zeros["conv.weight"] == 0)
    assert zeros["lif.v"] == 0.0
    assert zeros["lif.spike"] is None
    print("zeros_like_state: scalars/None preserved")

    print("\nALL AGGREGATION SELF-TESTS PASSED.")
