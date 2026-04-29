"""
Cloud-removal paired-image dataset + federated partitioning.

Supports the CUHK-CR1 / CUHK-CR2 benchmarks (Sui et al., TGRS 2024) and is
layout-compatible with RICE1/RICE2 (Lin et al., 2019).  Both layouts expose
paired "cloudy / clear" images on disk; we abstract over the on-disk
folder naming conventions.

Accepted folder layouts (auto-detected in `_discover_folders`):

    <root>/
        train/input/   *.png|jpg|tif          (cloudy)
        train/target/  *.png|jpg|tif          (clear)
        test/input/    *.png|jpg|tif
        test/target/   *.png|jpg|tif

    <root>/
        input/   *.png|...                    (all samples, no split)
        target/  *.png|...

    <root>/
        cloudy/  *.png|...                    (CUHK-CR upstream naming)
        clear/   *.png|...

    <root>/
        cloud/   *.png|...                    (seen in some mirrors)
        label/   *.png|...

The loader returns tensors of shape [3, H, W] in [0, 1], matching the
input contract of VLIFNet.forward().

v1 USE-CASES
------------
1. `PairedCloudDataset(root, split='train', patch_size=64)` — training
   loader that random-crops each image to `patch_size × patch_size`.

2. `PairedCloudDataset(root, split='test', patch_size=None)` — evaluation
   loader that returns full-resolution images.

3. `build_client_partitions(ds, num_clients=50, mode='iid', seed=0)` —
   returns a list of `torch.utils.data.Subset` objects, one per client.
   v1 only exercises `mode='iid'`; `mode='dirichlet_cluster'` is
   declared but raises NotImplementedError (v2 scope).

Every file is independently loaded with PIL to keep DataLoader worker
startup light, matching upstream dataset_load.py behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Folder-layout auto-discovery
# ---------------------------------------------------------------------------

_INPUT_NAMES  = ("input", "cloudy", "cloud")
_TARGET_NAMES = ("target", "clear", "label", "gt")
_IMAGE_EXTS   = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in _IMAGE_EXTS and p.is_file()])


def _pick(folder: Path, candidates: Sequence[str]) -> Optional[Path]:
    """Return the first existing sub-folder among `candidates`, else None."""
    for name in candidates:
        p = folder / name
        if p.is_dir():
            return p
    return None


def _discover_folders(root: Path, split: Optional[str]
                      ) -> Tuple[Path, Path]:
    """Resolve (input_dir, target_dir) given the dataset root and an
    optional split name ('train' / 'test' / None).

    Raises FileNotFoundError if no matching layout is found.
    """
    root = Path(root)
    bases: List[Path] = []
    if split:
        bases.append(root / split)
    bases.append(root)

    for base in bases:
        if not base.is_dir():
            continue
        in_dir  = _pick(base, _INPUT_NAMES)
        tar_dir = _pick(base, _TARGET_NAMES)
        if in_dir is not None and tar_dir is not None:
            return in_dir, tar_dir

    raise FileNotFoundError(
        f"Could not find paired input/target folders under '{root}' "
        f"(split={split!r}).  Tried names: input={_INPUT_NAMES}, "
        f"target={_TARGET_NAMES}."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _read_rgb(path: Path) -> np.ndarray:
    """Load an image as float32 HWC in [0, 1], forced to 3 channels."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


class PairedCloudDataset(Dataset):
    """Paired cloudy / clear image dataset with optional random-crop.

    Parameters
    ----------
    root : str | Path
        Dataset root directory.
    split : {'train', 'test', None}
        'train' / 'test' picks a sub-folder of that name if present;
        None assumes the paired folders live directly under `root`.
    patch_size : int | None
        Random square crop size during training.  Use `None` to return
        the full image (for evaluation / inference).
    pair_by : {'name', 'order'}
        - 'name' : match files by filename (recommended for CUHK-CR;
                   falls back to sorted-order if not all names match).
        - 'order': zip by sorted order (matches upstream dataset_load.py).
    """

    def __init__(self,
                 root: str,
                 split: Optional[str] = None,
                 patch_size: Optional[int] = 64,
                 pair_by: str = "name"):
        super().__init__()

        assert pair_by in ("name", "order")
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.pair_by = pair_by

        in_dir, tar_dir = _discover_folders(self.root, split)
        in_files  = _list_images(in_dir)
        tar_files = _list_images(tar_dir)

        if pair_by == "name":
            tar_by_name = {p.stem: p for p in tar_files}
            pairs = []
            for p in in_files:
                if p.stem in tar_by_name:
                    pairs.append((p, tar_by_name[p.stem]))
            if len(pairs) < min(len(in_files), len(tar_files)) * 0.9:
                # Too few name matches → assume unrelated naming, use order.
                pairs = list(zip(in_files, tar_files))
        else:
            pairs = list(zip(in_files, tar_files))

        assert len(pairs) > 0, f"No image pairs found in {in_dir} / {tar_dir}"
        self.pairs = pairs
        self.in_dir = in_dir
        self.tar_dir = tar_dir

    # -- torch.utils.data.Dataset protocol ---------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        in_path, tar_path = self.pairs[idx]
        cloudy = _read_rgb(in_path)   # HWC, [0, 1]
        clear  = _read_rgb(tar_path)

        h, w = cloudy.shape[:2]
        if self.patch_size is not None:
            # Random crop, shared between input & target
            p = self.patch_size
            assert h >= p and w >= p, (
                f"Image {in_path} is smaller ({h}x{w}) than patch_size={p}")
            r = np.random.randint(0, h - p + 1)
            c = np.random.randint(0, w - p + 1)
            cloudy = cloudy[r:r + p, c:c + p]
            clear  = clear[r:r + p, c:c + p]

        # HWC -> CHW, float32 tensor
        cloudy_t = torch.from_numpy(np.ascontiguousarray(cloudy.transpose(2, 0, 1)))
        clear_t  = torch.from_numpy(np.ascontiguousarray(clear.transpose(2, 0, 1)))
        return cloudy_t, clear_t

    # -- diagnostics -------------------------------------------------------

    def describe(self) -> str:
        return (f"PairedCloudDataset(root={self.root}, split={self.split}, "
                f"patch_size={self.patch_size}, pair_by={self.pair_by}, "
                f"n={len(self)}, input_dir={self.in_dir.name}, "
                f"target_dir={self.tar_dir.name})")


# ---------------------------------------------------------------------------
# Train / test split helper (when dataset on-disk has no explicit split)
# ---------------------------------------------------------------------------

def split_train_test(dataset: PairedCloudDataset,
                     test_ratio: float = 0.2,
                     seed: int = 0) -> Tuple[Subset, Subset]:
    """If the dataset has no train/test subfolders, call this to get
    Subsets for train / test according to `test_ratio`."""
    n = len(dataset)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_ratio)))
    test_idx = idx[:n_test].tolist()
    train_idx = idx[n_test:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


# ---------------------------------------------------------------------------
# Federated partitioning
# ---------------------------------------------------------------------------

def build_client_partitions(dataset: Dataset,
                            num_clients: int,
                            mode: str = "iid",
                            seed: int = 0,
                            min_per_client: int = 1,
                            ) -> List[Subset]:
    """Split `dataset` into `num_clients` disjoint Subsets.

    Parameters
    ----------
    mode : {'iid', 'dirichlet_cluster'}
        'iid'               — random permutation, equal-sized splits.
                              This is the v1 default.
        'dirichlet_cluster' — placeholder for v2; k-means the images on
                              a pretrained feature extractor, then
                              Dirichlet-distribute cluster labels.

    Guarantees every client receives at least `min_per_client` samples.
    Extra samples (len(ds) % num_clients) go to the first few clients.
    """
    n = len(dataset)
    if mode == "iid":
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        base = n // num_clients
        rem  = n % num_clients
        if base < min_per_client:
            raise ValueError(
                f"dataset size {n} too small for {num_clients} clients "
                f"(needs ≥ {num_clients * min_per_client} samples)")
        subsets: List[Subset] = []
        cursor = 0
        for c in range(num_clients):
            size = base + (1 if c < rem else 0)
            idx  = perm[cursor:cursor + size].tolist()
            subsets.append(Subset(dataset, idx))
            cursor += size
        return subsets

    if mode == "dirichlet_cluster":
        # v2 scope.  Kept here so callers can pick the mode already.
        raise NotImplementedError(
            "Dirichlet-cluster partitioning is scheduled for v2. "
            "In v1 please use mode='iid'.")

    raise ValueError(f"Unknown partition mode: {mode}")


# ---------------------------------------------------------------------------
# Structured planes-of-satellites helper (matches FLSNN's plane/sat layout)
# ---------------------------------------------------------------------------

def build_plane_satellite_partitions(dataset: Dataset,
                                     num_planes: int,
                                     sats_per_plane: int,
                                     mode: str = "iid",
                                     seed: int = 0,
                                     ) -> List[List[Subset]]:
    """Two-level partition: outer list = planes, inner list = satellites.

    Useful for the FLSNN-style constellation that addresses satellites as
    `(plane_idx, sat_idx)`.  The total number of clients is
    `num_planes * sats_per_plane`.

    Returns a list-of-lists of Subset objects.
    """
    total_clients = num_planes * sats_per_plane
    flat = build_client_partitions(dataset, total_clients, mode=mode, seed=seed)
    grouped: List[List[Subset]] = []
    k = 0
    for _ in range(num_planes):
        grouped.append(flat[k:k + sats_per_plane])
        k += sats_per_plane
    return grouped


# ---------------------------------------------------------------------------
# Quick sanity check (run directly: python cloud_removal_dataset.py <root>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./data/CUHK-CR1"
    print(f"Probing {root} ...")

    # Try train split first; if that fails, try flat layout.
    try:
        train = PairedCloudDataset(root, split="train", patch_size=64)
        test  = PairedCloudDataset(root, split="test",  patch_size=None)
        print("train:", train.describe())
        print("test: ", test.describe())
    except FileNotFoundError:
        ds = PairedCloudDataset(root, split=None, patch_size=64)
        print("flat: ", ds.describe())
        tr, te = split_train_test(ds, test_ratio=0.2)
        print(f"derived split: |train|={len(tr)}  |test|={len(te)}")

        parts = build_plane_satellite_partitions(tr, num_planes=5,
                                                 sats_per_plane=10, mode="iid")
        for i, plane in enumerate(parts):
            sizes = [len(s) for s in plane]
            print(f"  plane {i}: sizes={sizes}  total={sum(sizes)}")

    # Sample one item
    x, y = ds[0] if 'ds' in dir() else train[0]
    print("shapes:", x.shape, y.shape, "dtype:", x.dtype,
          "range:", x.min().item(), x.max().item())
