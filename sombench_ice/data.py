import math
import re
from pathlib import Path

import numpy as np

from .common import read_json, sha256, signature, write_json
from .core.schema import CHANNELS, GROUP_INDICES, LAYERS, SCHEMA_ID

ARRAYS = {"pixels": (9, "float32"), "valid": (9, "bool"),
          "target": (1, "float32"), "target_valid": (1, "bool")}


class Moments:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0., 0.

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        n = values.size
        if not n:
            return
        mean = float(values.mean())
        m2 = float(np.square(values - mean).sum())
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.n * n / (self.n + n)
        self.mean += delta * n / (self.n + n)
        self.n += n

    def result(self):
        if not self.n:
            raise ValueError("An input channel has no finite valid training pixels")
        std = math.sqrt(max(0., self.m2 / self.n))
        return {"count": self.n, "mean": self.mean, "std": std,
                "scale": std if std > 1e-8 else 1., "constant": std <= 1e-8}


def split_lists(root):
    result, seen = {}, set()
    for split in ("train", "val", "test"):
        path = root / f"{split}_filtered.txt"
        ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Empty split or duplicate IDs in {path}")
        if any(not re.fullmatch(r"patch_\d+_\d+(?:_S)?_80[NS]", name) for name in ids):
            raise ValueError(f"Expected official complete patch IDs in {path}")
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Split leakage: repeated parent IDs {sorted(overlap)[:5]}")
        seen.update(ids)
        result[split] = ids
    return result


def read_patch(root, patch_id):
    import rasterio
    arrays, masks, hashes, reference = [], [], {}, None
    for suffix, count in (*LAYERS, ("PRO", 1)):
        path = root / f"{patch_id}_{suffix}.tif"
        with rasterio.open(path) as src:
            if src.count != count:
                raise ValueError(f"{path.name}: expected {count} bands, found {src.count}; "
                                 "check the published aspect encoding before proceeding")
            grid = (src.height, src.width, src.crs, src.transform)
            if reference is None:
                reference = grid
            if grid != reference:
                raise ValueError(f"Unaligned pixel grid/CRS: {path.name}; no automatic resampling")
            if src.crs is None:
                raise ValueError(f"Missing CRS: {path}")
            raw = src.read(out_dtype="float32")
            valid = (src.read_masks() > 0) & np.isfinite(raw)
            scales = np.asarray(src.scales, dtype=np.float32)[:, None, None]
            offsets = np.asarray(src.offsets, dtype=np.float32)[:, None, None]
            values = raw * scales + offsets
            valid &= np.isfinite(values)
            arrays.append(np.where(valid, values, 0.).astype(np.float32))
            masks.append(valid)
        hashes[path.name] = sha256(path)
    pixels, valid = np.concatenate(arrays[:-1]), np.concatenate(masks[:-1])
    target, target_valid = arrays[-1], masks[-1]
    if np.any((target[target_valid] < 0.) | (target[target_valid] > 1.)):
        raise ValueError(f"{patch_id}: valid PRO values outside [0,1]; check nodata/scale metadata")
    height, width, crs, transform = reference
    grid_info = {"height": height, "width": width, "crs": crs.to_wkt(),
                 "transform": list(transform)[:6]}
    return pixels, valid, target, target_valid, hashes, grid_info


def prepare(data_dir, cache_dir, patch_size=256):
    import rasterio
    if patch_size < 32 or patch_size % 32:
        raise ValueError("patch_size must be a positive multiple of 32")
    root, output = Path(data_dir).resolve(), Path(cache_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("Keep the prepared cache outside the raw-data directory")
    splits = split_lists(root)
    # Validate file presence and determine storage size before creating anything.
    plan = []
    for split, ids in splits.items():
        for patch_id in ids:
            for suffix, _ in (*LAYERS, ("PRO", 1)):
                path = root / f"{patch_id}_{suffix}.tif"
                if not path.is_file():
                    raise FileNotFoundError(path)
            with rasterio.open(root / f"{patch_id}_PRO.tif") as src:
                plan.append((split, patch_id, src.height, src.width))
    total = sum(math.ceil(h / patch_size) * math.ceil(w / patch_size) for _, _, h, w in plan)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing dataset.
    shape = (total, 1, patch_size, patch_size)
    stores = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype,
              shape=(shape[0], channels, *shape[2:])) for name, (channels, dtype) in ARRAYS.items()}
    statistics = [Moments() for _ in CHANNELS]
    records, sources, parent_grids = [], {}, {}
    index = 0
    for number, (split, patch_id, height, width) in enumerate(plan, 1):
        pixels, valid, target, target_valid, hashes, grid = read_patch(root, patch_id)
        if pixels.shape[-2:] != (height, width):
            raise ValueError("Input changed during preparation")
        sources.update(hashes)
        parent_grids[patch_id] = grid
        if split == "train":
            for channel, moments in enumerate(statistics):
                moments.add(pixels[channel][valid[channel]])
        for row in range(0, height, patch_size):
            for col in range(0, width, patch_size):
                h, w = min(patch_size, height-row), min(patch_size, width-col)
                for name, array in (("pixels", pixels), ("valid", valid),
                                    ("target", target), ("target_valid", target_valid)):
                    stores[name][index] = 0
                    stores[name][index, :, :h, :w] = array[:, row:row+h, col:col+w]
                n_valid = int(stores["target_valid"][index].sum())
                if n_valid and not stores["valid"][index].any():
                    raise ValueError(f"{patch_id}@{row},{col}: valid target but no observed input")
                records.append({"index": index, "split": split, "parent_id": patch_id,
                                "row": row, "col": col, "height": h, "width": w,
                                "valid_targets": n_valid})
                index += 1
        if number == 1 or number % 10 == 0 or number == len(plan):
            print(f"prepare {number}/{len(plan)} parent patches; {index}/{total} tiles", flush=True)
    for array in stores.values():
        array.flush()
    for split in splits:
        sources[f"{split}_filtered.txt"] = sha256(root / f"{split}_filtered.txt")
    stats = {name: values.result() for name, values in zip(CHANNELS, statistics)}
    files = {f"{name}.npy": sha256(output / f"{name}.npy") for name in ARRAYS}
    metadata = {"schema": SCHEMA_ID, "channels": list(CHANNELS), "groups": GROUP_INDICES,
                "patch_size": patch_size, "splits": splits, "records": records,
                "parent_grids": parent_grids, "statistics": stats, "sources": sources,
                "arrays_sha256": files, "normalization_scope": "train_valid_pixels_only",
                "tiling": "nonoverlapping_top_left_grid_with_invalid_edge_padding",
                "metric_mask": "finite_valid_PRO_pixels_including_valid_zero_values"}
    metadata["signature"] = signature(metadata)
    write_json(output / "metadata.json", metadata)
    summary = {split: {"parents": len(ids),
                      "tiles": sum(r["split"] == split and r["valid_targets"] > 0 for r in records),
                      "valid_pixels": sum(r["valid_targets"] for r in records if r["split"] == split)}
               for split, ids in splits.items()}
    if any(row["valid_pixels"] == 0 for row in summary.values()):
        raise ValueError("At least one official split has no valid target pixels")
    write_json(output / "summary.json", summary)
    print(f"Prepared cache: {output}\nSplit counts: {summary}", flush=True)
    return metadata


def read_metadata(cache_dir, verify=True):
    root = Path(cache_dir)
    meta = read_json(root / "metadata.json")
    expected = meta.pop("signature")
    if signature(meta) != expected or meta["schema"] != SCHEMA_ID:
        raise ValueError("Cache metadata/schema mismatch; prepare again in a new directory")
    meta["signature"] = expected
    if verify:
        for name, digest in meta["arrays_sha256"].items():
            if sha256(root / name) != digest:
                raise ValueError(f"Prepared array checksum mismatch: {name}")
    return meta


class IceDataset:
    def __init__(self, cache_dir, split, metadata):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        self.root, self.meta = Path(cache_dir), metadata
        self.rows = [r for r in metadata["records"] if r["split"] == split and r["valid_targets"] > 0]
        if not self.rows:
            raise ValueError(f"No supervised tiles in {split}")
        self.mean = np.array([metadata["statistics"][x]["mean"] for x in CHANNELS], dtype=np.float32)[:, None, None]
        self.scale = np.array([metadata["statistics"][x]["scale"] for x in CHANNELS], dtype=np.float32)[:, None, None]
        self._arrays = None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        return {**self.__dict__, "_arrays": None}

    def __getitem__(self, item):
        if self._arrays is None:
            self._arrays = {name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                            for name in ARRAYS}
        row = self.rows[item]
        result = {name: np.array(array[row["index"]], copy=True) for name, array in self._arrays.items()}
        result["pixels"] = np.where(result["valid"], (result["pixels"] - self.mean) / self.scale, 0.).astype(np.float32)
        result["tile_index"] = row["index"]
        return result
