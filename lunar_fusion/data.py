import json
import math
import mmap
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import CHANNELS, GROUP_INDICES, REGRESSION_CHANNELS


def _advise_random_mapping(array):
    """Optional OS access hint; does not modify file contents or sampled values.

    Global maps are read as short rows separated by large strides. Linux's
    normal readahead can fetch many unused pages under a tight memory cgroup.
    MADV_RANDOM is a performance experiment, not a guaranteed speedup.
    """
    mapping = getattr(array, "_mmap", None)
    if not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_RANDOM"):
        raise RuntimeError(
            "runtime.mmap_read_hint=random requires mmap.madvise/MADV_RANDOM "
            "(available on Linux). Use runtime.mmap_read_hint=default here.")
    try:
        mapping.madvise(mmap.MADV_RANDOM)
    except OSError as error:
        raise RuntimeError("The OS rejected MADV_RANDOM; use mmap_read_hint=default") from error


class MoonstoneStore:
    def __init__(self, directory, allow_missing_channels=False, read_hint="default"):
        if read_hint not in ("default", "random"):
            raise ValueError("read_hint must be default or random")
        self.read_hint = read_hint
        directory = Path(directory)
        if not (directory / "channel_index.json").is_file():
            directory = directory / "mmap"
        index_path = directory / "channel_index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Cannot find channel_index.json under {directory}")
        self.directory = directory
        with index_path.open(encoding="utf-8") as handle:
            self.index = json.load(handle)
        if self.index.get("normalized") is not True:
            raise ValueError("Expected pre-normalized Moonstone arrays; normalization is not applied twice")
        if self.index.get("dtype") != "float32":
            raise ValueError("Expected float32 data")
        self.shape = tuple(self.index["shape"])
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("Invalid global array shape")
        self.paths = {}
        self.formats = {}
        self.mask_paths = {}
        for name in CHANNELS:
            path = directory / f"{name}.npy"
            mask_path = directory / f"{name}_mask.npy"
            if not path.is_file():
                if allow_missing_channels:
                    continue
                raise FileNotFoundError(f"Required channel missing: {path}")
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Validity mask missing: {mask_path}. Refusing to infer validity from value != 0.")
            with path.open("rb") as handle:
                normal_npy = handle.read(6) == b"\x93NUMPY"
            if normal_npy:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != self.shape or array.dtype != np.float32:
                    raise ValueError(f"Unexpected shape/dtype: {path}")
                del array
            elif path.stat().st_size != math.prod(self.shape) * 4:
                raise ValueError(f"Wrong byte size / incomplete download: {path}")
            packed = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if packed.dtype != np.uint8 or packed.ndim != 1 or packed.size != (math.prod(self.shape) + 7) // 8:
                raise ValueError(f"Expected np.packbits uint8 nodata mask: {mask_path}")
            del packed
            self.paths[name] = path
            self.mask_paths[name] = mask_path
            self.formats[name] = normal_npy
        if not self.paths:
            raise ValueError("No input channels found")
        self._arrays, self._masks = {}, {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"], state["_masks"] = {}, {}
        return state

    def _open(self, name):
        if name not in self._arrays:
            array = (
                np.load(self.paths[name], mmap_mode="r", allow_pickle=False)
                if self.formats[name] else
                np.memmap(self.paths[name], mode="r", dtype=np.float32, shape=self.shape))
            mask = np.load(self.mask_paths[name], mmap_mode="r", allow_pickle=False)
            if self.read_hint == "random":
                _advise_random_mapping(array)
                _advise_random_mapping(mask)
            self._arrays[name], self._masks[name] = array, mask
        return self._arrays[name], self._masks[name]

    def read(self, row, col, size):
        h, w = self.shape
        if row < 0 or col < 0 or row + size > h or col + size > w:
            raise IndexError(f"Crop {(row, col, size)} outside global shape {self.shape}")
        pixels = np.zeros((len(CHANNELS), size, size), dtype=np.float32)
        valid = np.zeros_like(pixels, dtype=bool)
        # Packed masks use np.packbits(..., bitorder='big') in the source pipeline.
        flat = (row + np.arange(size, dtype=np.int64))[:, None] * w
        flat = flat + (col + np.arange(size, dtype=np.int64))[None, :]
        byte_ids, shifts = flat // 8, 7 - flat % 8
        for channel, name in enumerate(CHANNELS):
            if name not in self.paths:
                continue
            array, packed = self._open(name)
            values = np.asarray(array[row:row + size, col:col + size])
            missing = ((packed[byte_ids] >> shifts) & 1).astype(bool)
            available = ~missing & np.isfinite(values)
            pixels[channel] = np.where(available, values, 0)
            valid[channel] = available
        return {"pixels": torch.from_numpy(pixels), "valid": torch.from_numpy(valid),
                "row": row, "col": col}

    def close(self):
        self._arrays.clear()
        self._masks.clear()


def read_metadata(path, protocol="stored"):
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Install h5py to read benchmark split metadata") from error
    with h5py.File(path, "r") as handle:
        meta = handle["metadata"]
        result = {key: np.asarray(meta[key]) for key in
                  ("row_idx", "col_idx", "center_lat", "has_channel", "split")}
        result["unit_ids"] = np.asarray(meta["geology/dominant_unit_id"])
        result["unit_names"] = np.asarray([
            name.decode() if isinstance(name, bytes) else str(name)
            for name in meta["geology/dominant_unit_name"][:]])
        result["patch_size"] = int(handle.attrs.get("patch_size", 256))
        names = [name.decode() if isinstance(name, bytes) else str(name)
                 for name in handle.attrs["channel_names"]]
        if names != CHANNELS:
            raise ValueError("HDF5 channel order does not match Moonstone V4")
    if protocol == "latitude":
        latitude = np.abs(result["center_lat"])
        split = np.full(len(latitude), -1, dtype=np.int8)
        split[latitude <= 60] = 0
        split[(latitude > 60) & (latitude <= 70)] = 1
        split[(latitude > 70) & (latitude <= 80)] = 2
        result["split"] = split
    return result


class RandomCropDataset(Dataset):
    """Deterministic (seed,epoch,index) random crops fully contained in one split.

    This constrains ALL benchmark cells touched by a crop, not only its center.
    The 'all' training pool is explicitly transductive and may overlap val/test.
    Fixed validation crops use an unchanging epoch and a separate seed.
    """
    def __init__(self, data_cfg, split="train", seed=42, *, mmap_read_hint="default"):
        self.cfg = data_cfg.copy()
        self.seed, self.epoch = int(seed), 0
        self.split = split
        self.store = MoonstoneStore(data_cfg["mmap_dir"], data_cfg["allow_missing_channels"],
                                    read_hint=mmap_read_hint)
        self.size = int(data_cfg["patch_size"])
        self.length = int(data_cfg["train_samples"] if split == "train" else data_cfg["val_samples"])
        ppd = data_cfg["ppd"]
        south, north = data_cfg["latitude_range"]
        self.row_min = max(0, math.ceil((90 - north) * ppd))
        self.row_max = min(self.store.shape[0] - self.size,
                           math.floor((90 - south) * ppd) - self.size)
        if self.row_min > self.row_max or self.size > self.store.shape[1]:
            raise ValueError("Crop/latitude range does not fit the global grid")
        self.grid = None
        if split != "train" or data_cfg["pretrain_pool"] == "train":
            meta = read_metadata(data_cfg["h5_path"], data_cfg["split_protocol"])
            self.cell_size = meta["patch_size"]
            shape = (math.ceil(self.store.shape[0] / self.cell_size),
                     math.ceil(self.store.shape[1] / self.cell_size))
            self.grid = np.full(shape, -1, dtype=np.int8)
            self.grid[meta["row_idx"], meta["col_idx"]] = meta["split"]
            self.split_id = 0 if split == "train" else 1
            self.cells = np.argwhere(self.grid == self.split_id)
            tops = self.cells[:, 0] * self.cell_size
            self.cells = self.cells[(tops >= self.row_min) & (tops <= self.row_max)]
            if not len(self.cells):
                raise ValueError("No cells satisfy this split/latitude range")
            if self.size > self.cell_size:
                raise ValueError("Split-constrained random crops must not exceed benchmark patch_size")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.length

    def _inside_split(self, row, col):
        if self.grid is None:
            return True
        s = self.cell_size
        touched = self.grid[row // s:(row + self.size - 1) // s + 1,
                            col // s:(col + self.size - 1) // s + 1]
        return bool((touched == self.split_id).all())

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        for attempt in range(100):
            if self.grid is None:
                row = int(rng.integers(self.row_min, self.row_max + 1))
                col = int(rng.integers(0, self.store.shape[1] - self.size + 1))
            else:
                cell = self.cells[int(rng.integers(len(self.cells)))]
                # Alternate broad offset crops with safe crops inside one cell.
                max_offset = self.cell_size if attempt % 2 == 0 else self.cell_size - self.size + 1
                row = int(cell[0] * self.cell_size + rng.integers(max_offset))
                col = int(cell[1] * self.cell_size + rng.integers(max_offset))
                if row > self.row_max or col + self.size > self.store.shape[1]:
                    continue
                if not self._inside_split(row, col):
                    continue
            sample = self.store.read(row, col, self.size)
            if float(sample["valid"][0].float().mean()) < self.cfg["min_surface_fraction"]:
                continue
            if not sample["valid"].any():
                continue
            return sample
        raise RuntimeError(
            "No valid crop after 100 attempts. Check masks, split, latitude range, "
            "min_surface_fraction and data completeness.")


def age_from_name(name):
    for prefix, label in (("pN", 0), ("EI", 2), ("N", 1), ("I", 2), ("E", 3), ("C", 4)):
        if name.startswith(prefix):
            return label
    return -1


def make_benchmark_store(data_cfg):
    """Choose the pixel source; task labels and split selection stay unchanged."""
    if data_cfg.get("benchmark_backend", "mmap") == "hdf5":
        from .hdf5_store import HDF5PatchStore
        return HDF5PatchStore(data_cfg)
    return MoonstoneStore(data_cfg["mmap_dir"], data_cfg["allow_missing_channels"])


class BenchmarkDataset(Dataset):
    """All six tasks share fixed metadata splits, including thermal regression."""
    def __init__(self, data_cfg, task, split, drop_groups=(), few_shot=0, seed=42):
        self.task, self.drop_groups = task, tuple(drop_groups)
        self.store = make_benchmark_store(data_cfg)
        self.meta = read_metadata(data_cfg["h5_path"], data_cfg["split_protocol"])
        self.size = self.meta["patch_size"]
        if data_cfg["patch_size"] != self.size:
            raise ValueError("Downstream tasks preserve the released benchmark patch size")
        split_id = {"train": 0, "val": 1, "test": 2}[split]
        keep = self.meta["split"] == split_id
        self.ages = np.array([age_from_name(name) for name in self.meta["unit_names"]])
        if task == "geology":
            keep &= (self.meta["unit_ids"] >= 1) & (self.meta["unit_ids"] <= 49)
        if task == "age":
            keep &= self.ages >= 0
        if task in REGRESSION_CHANNELS:
            indices = [CHANNELS.index(name) for name in REGRESSION_CHANNELS[task]]
            keep &= self.meta["has_channel"][:, indices].any(axis=1)
        self.indices = np.where(keep)[0]
        if few_shot:
            if task not in ("geology", "age"):
                raise ValueError("few_shot means K per class and is supported for geology/age only")
            if split == "train":
                labels = self.meta["unit_ids"] - 1 if task == "geology" else self.ages
                rng, chosen = np.random.default_rng(seed), []
                for label in np.unique(labels[self.indices]):
                    candidates = self.indices[labels[self.indices] == label]
                    chosen.extend(rng.choice(candidates, min(few_shot, len(candidates)), replace=False))
                self.indices = np.asarray(sorted(chosen))
        if not len(self.indices):
            raise ValueError(f"Empty {task}/{split} split")
        self.label_path = None
        self._label = None
        if task in ("mare", "craters"):
            name = "mare_mask.tif" if task == "mare" else "crater_mask.tif"
            self.label_path = Path(data_cfg["label_dir"]) / name
            if not self.label_path.is_file():
                raise FileNotFoundError(self.label_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_label"] = None
        return state

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real = self.indices[index]
        row = int(self.meta["row_idx"][real]) * self.size
        col = int(self.meta["col_idx"][real]) * self.size
        sample = self.store.read(row, col, self.size)
        pixels, valid = sample["pixels"], sample["valid"]
        if self.task in ("geology", "age"):
            label = int(self.meta["unit_ids"][real]) - 1 if self.task == "geology" else int(self.ages[real])
            target, target_valid = torch.tensor(label), torch.tensor(True)
        elif self.task in REGRESSION_CHANNELS:
            values, availability = [], []
            for name in REGRESSION_CHANNELS[self.task]:
                channel = CHANNELS.index(name)
                mask = valid[channel]
                availability.append(bool(mask.any()))
                values.append(pixels[channel][mask].mean() if mask.any() else torch.tensor(0.0))
                # Prevent target leakage into the encoder, including its validity input.
                pixels[channel] = 0
                valid[channel] = False
            target = torch.stack(values)
            target_valid = torch.tensor(availability, dtype=torch.bool)
        else:
            import rasterio
            if self._label is None:
                self._label = rasterio.open(self.label_path)
                if self._label.shape != self.store.shape:
                    raise ValueError("Label raster and input grid shapes differ")
            window = rasterio.windows.Window(col, row, self.size, self.size)
            labels = self._label.read(1, window=window)
            # Moonstone label zero denotes valid background, not a missing observation.
            target = torch.from_numpy(labels.astype(np.int64))
            target_valid = (target == 0) | (target == 1)
        for group in self.drop_groups:
            pixels[list(GROUP_INDICES[group])] = 0
            valid[list(GROUP_INDICES[group])] = False
        sample.update(target=target, target_valid=target_valid, patch_id=int(real))
        return sample

    def close(self):
        self.store.close()
        if self._label is not None:
            self._label.close()
            self._label = None


class FixedPatchDataset(Dataset):
    """Released patch coordinates for held-out reconstruction evaluation."""
    def __init__(self, data_cfg, split):
        self.store = make_benchmark_store(data_cfg)
        self.meta = read_metadata(data_cfg["h5_path"], data_cfg["split_protocol"])
        self.size = self.meta["patch_size"]
        if data_cfg["patch_size"] != self.size:
            raise ValueError("Fixed-patch evaluation requires the released patch size")
        self.indices = np.flatnonzero(self.meta["split"] == {"train": 0, "val": 1, "test": 2}[split])
        if not len(self.indices):
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real = self.indices[index]
        return self.store.read(int(self.meta["row_idx"][real]) * self.size,
                               int(self.meta["col_idx"][real]) * self.size, self.size)

    def close(self):
        self.store.close()
