import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import MoonstoneStore
from .schema import CHANNELS


class HDF5PatchStore(MoonstoneStore):
    def __init__(self, data_cfg):
        # Retain original grid / channel availability and provenance. This
        # opens only file headers, never the raw global pixel windows.
        super().__init__(data_cfg["mmap_dir"], data_cfg["allow_missing_channels"])
        self.h5_path = Path(data_cfg["h5_path"])
        stats_path = data_cfg.get("normalization_stats")
        if not stats_path:
            raise ValueError("HDF5 input requires the normalization statistics used for mmap")
        encoded = Path(stats_path).read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        expected = data_cfg.get("normalization_stats_sha256")
        if expected and digest != expected:
            raise ValueError("Normalization statistics changed")
        statistics = json.loads(encoded.decode("utf-8-sig"))
        self.statistics = []
        for name in CHANNELS:
            if name not in statistics or len(statistics[name]) != 2:
                raise ValueError(f"Missing [mean, std] statistics for {name}")
            mean, std = map(float, statistics[name])
            if not np.isfinite([mean, std]).all() or std < 0:
                raise ValueError(f"Invalid normalization statistics for {name}")
            self.statistics.append((mean, std))
        with h5py.File(self.h5_path, "r") as handle:
            if "patches" not in handle:
                raise ValueError("This HDF5 contains no patches dataset; use the complete benchmark file")
            if bool(handle.attrs.get("normalized", False)):
                raise ValueError("Expected unnormalized released HDF5 patches")
            patches = handle["patches"]
            self.size = int(handle.attrs.get("patch_size", 256))
            names = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in handle.attrs["channel_names"]]
            if names != CHANNELS or patches.shape[1:] != (len(CHANNELS), self.size, self.size):
                raise ValueError("HDF5 patch channel order or shape differs from Moonstone V4")
            if patches.dtype != np.dtype("float32") or data_cfg["patch_size"] != self.size:
                raise ValueError("HDF5 input requires float32 patches at the benchmark patch size")
            rows = np.asarray(handle["metadata/row_idx"], dtype=np.int64)
            cols = np.asarray(handle["metadata/col_idx"], dtype=np.int64)
            if rows.shape != (len(patches),) or cols.shape != rows.shape:
                raise ValueError("HDF5 coordinates do not match its patches")
            self.positions = {(int(r) * self.size, int(c) * self.size): i
                              for i, (r, c) in enumerate(zip(rows, cols))}
            if len(self.positions) != len(rows):
                raise ValueError("Duplicate benchmark coordinates cannot identify HDF5 patches")
        self._h5, self._h5_pid = None, None

    def __getstate__(self):
        state = super().__getstate__()
        state["_h5"], state["_h5_pid"] = None, None
        return state

    def _open_h5(self):
        # Each multiprocessing worker gets an independent read-only handle.
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.h5_path, "r", rdcc_nbytes=8 * 1024 ** 2)
            self._h5_pid = pid
        return self._h5

    def read(self, row, col, size):
        if size != self.size or (row, col) not in self.positions:
            raise ValueError("HDF5 input only supports the original fixed benchmark windows")
        handle = self._open_h5()
        pixels = handle["patches"][self.positions[(row, col)]].astype(np.float32, copy=False)
        # Released HDF5 retains NaN/Inf; a finite measurement of zero is valid.
        valid = np.isfinite(pixels)
        pixels[~valid] = 0.0
        for channel, name in enumerate(CHANNELS):
            if name not in self.paths:
                pixels[channel] = 0.0
                valid[channel] = False
                continue
            mean, std = self.statistics[channel]
            if std > 0:
                # Keep the same float32 operation order as step15_build_mmap.py.
                pixels[channel] = (pixels[channel] - mean) / std
        pixels[~valid] = 0.0
        # Mirrors MoonstoneStore's finite check after normalization.
        valid &= np.isfinite(pixels)
        pixels[~valid] = 0.0
        return {"pixels": torch.from_numpy(pixels), "valid": torch.from_numpy(valid),
                "row": row, "col": col}

    def close(self):
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
        self._h5, self._h5_pid = None, None
        super().close()
