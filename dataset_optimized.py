import os
import glob
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _is_hwc_shape(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16


def _is_chw_shape(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16


def ensure_image_chw(img: np.ndarray, expected_channels: Optional[int] = None) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"图像应为3维数组，实际 shape={img.shape}")

    if _is_hwc_shape(img):
        img = np.transpose(img, (2, 0, 1))
    elif _is_chw_shape(img):
        pass
    else:
        raise ValueError(f"无法识别图像通道位置，shape={img.shape}")

    if expected_channels is not None and img.shape[0] != expected_channels:
        raise ValueError(f"图像通道数不匹配，期望 {expected_channels}，实际 {img.shape[0]}，shape={img.shape}")
    return img


def ensure_mask_1hw(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]
    elif mask.ndim == 3:
        if mask.shape[0] == 1:
            pass
        elif mask.shape[-1] == 1:
            mask = np.transpose(mask, (2, 0, 1))
        else:
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(f"mask 无法压缩到二维，shape={mask.shape}")
            mask = mask[np.newaxis, :, :]
    else:
        raise ValueError(f"mask 维度不支持，shape={mask.shape}")
    return mask


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)
    return (mask > 0).astype(np.float32)


def parse_patch_name(path_or_name: str) -> Tuple[str, int, int]:
    name = os.path.basename(path_or_name)
    stem, ext = os.path.splitext(name)
    if ext.lower() != ".npy":
        raise ValueError(f"仅支持 .npy 文件名，当前: {name}")

    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "fan":
        raise ValueError(f"无法从文件名解析 fan_id/坐标: {name}")

    fan_id = f"{parts[0]}_{parts[1]}"
    row = int(parts[2])
    col = int(parts[3])
    return fan_id, row, col


@dataclass
class SampleRecord:
    img_path: str
    mask_path: str
    fan_id: str
    row: int
    col: int
    fg_ratio: float
    is_hard_negative: bool = False


def _build_minmax_stats(records: List[SampleRecord], expected_channels: int) -> Dict[str, Dict[str, np.ndarray]]:
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for rec in records:
        img = ensure_image_chw(np.load(rec.img_path), expected_channels=expected_channels).astype(np.float32)
        cur_min = img.min(axis=(1, 2))
        cur_max = img.max(axis=(1, 2))
        if rec.fan_id not in stats:
            stats[rec.fan_id] = {"min": cur_min.copy(), "max": cur_max.copy()}
        else:
            stats[rec.fan_id]["min"] = np.minimum(stats[rec.fan_id]["min"], cur_min)
            stats[rec.fan_id]["max"] = np.maximum(stats[rec.fan_id]["max"], cur_max)
    return stats


def _build_zscore_stats(records: List[SampleRecord], expected_channels: int) -> Dict[str, Dict[str, np.ndarray]]:
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for rec in records:
        img = ensure_image_chw(np.load(rec.img_path), expected_channels=expected_channels).astype(np.float64)
        c, h, w = img.shape
        flat = img.reshape(c, -1)
        if rec.fan_id not in stats:
            stats[rec.fan_id] = {
                "sum": flat.sum(axis=1),
                "sq_sum": (flat ** 2).sum(axis=1),
                "count": np.array([flat.shape[1]], dtype=np.int64),
            }
        else:
            stats[rec.fan_id]["sum"] += flat.sum(axis=1)
            stats[rec.fan_id]["sq_sum"] += (flat ** 2).sum(axis=1)
            stats[rec.fan_id]["count"] += flat.shape[1]

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for fan_id, v in stats.items():
        count = float(v["count"][0])
        mean = v["sum"] / count
        var = np.maximum(v["sq_sum"] / count - mean ** 2, 0.0)
        std = np.sqrt(var)
        out[fan_id] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return out


def normalize_image(
    img: np.ndarray,
    mode: str = "per_channel_minmax",
    eps: float = 1e-6,
    fan_stats: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    img = img.astype(np.float32)
    out = img.copy()

    if mode == "none":
        return out

    if mode == "per_channel_minmax":
        for c in range(out.shape[0]):
            ch = out[c]
            ch_min = ch.min()
            ch_max = ch.max()
            if ch_max > ch_min:
                out[c] = (ch - ch_min) / (ch_max - ch_min + eps)
            else:
                out[c] = 0.0
        return out

    if mode == "per_channel_zscore":
        for c in range(out.shape[0]):
            ch = out[c]
            mean = ch.mean()
            std = ch.std()
            out[c] = (ch - mean) / (std + eps) if std > eps else (ch - mean)
        return out

    if mode == "fan_channel_minmax":
        if fan_stats is None:
            raise ValueError("fan_channel_minmax 需要 fan_stats")
        ch_min = fan_stats["min"]
        ch_max = fan_stats["max"]
        for c in range(out.shape[0]):
            denom = ch_max[c] - ch_min[c]
            if denom > eps:
                out[c] = (out[c] - ch_min[c]) / (denom + eps)
            else:
                out[c] = 0.0
        return out

    if mode == "fan_channel_zscore":
        if fan_stats is None:
            raise ValueError("fan_channel_zscore 需要 fan_stats")
        mean = fan_stats["mean"].reshape(-1, 1, 1)
        std = fan_stats["std"].reshape(-1, 1, 1)
        out = (out - mean) / (std + eps)
        return out

    raise ValueError(f"不支持的 normalize mode: {mode}")


class FanPatchDataset(Dataset):
    """
    增强版冲积扇 patch 数据集

    优先使用 dataset_root/meta/samples.csv 中已有的 hard negative 标记；
    如果元数据里没有该字段，再回退到基于坐标邻接的自动推断。
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        expected_channels: int = 6,
        normalize_mode: str = "per_channel_minmax",
        return_path: bool = False,
        use_hard_negative: bool = True,
        hard_negative_radius_steps: int = 1,
        hard_negative_repeat: int = 2,
        positive_min_ratio: float = 1e-4,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.expected_channels = expected_channels
        self.normalize_mode = normalize_mode
        self.return_path = return_path
        self.use_hard_negative = bool(use_hard_negative and split == "train")
        self.hard_negative_radius_steps = max(1, int(hard_negative_radius_steps))
        self.hard_negative_repeat = max(1, int(hard_negative_repeat))
        self.positive_min_ratio = float(positive_min_ratio)

        self.image_dir = os.path.join(dataset_root, split, "images")
        self.mask_dir = os.path.join(dataset_root, split, "masks")
        self.meta_csv = os.path.join(dataset_root, "meta", "samples.csv")

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"掩膜目录不存在: {self.mask_dir}")

        self.records = self._collect_records()
        if len(self.records) == 0:
            raise RuntimeError(f"{split} 集没有找到可用样本，请检查目录: {self.image_dir}")

        self.fan_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        if self.normalize_mode == "fan_channel_minmax":
            print(f"[FanPatchDataset] 正在构建 fan 级 minmax 统计: split={split}")
            self.fan_stats = _build_minmax_stats(self.records, expected_channels=self.expected_channels)
        elif self.normalize_mode == "fan_channel_zscore":
            print(f"[FanPatchDataset] 正在构建 fan 级 zscore 统计: split={split}")
            self.fan_stats = _build_zscore_stats(self.records, expected_channels=self.expected_channels)

        if self.use_hard_negative:
            self.samples = self._build_train_samples_with_hard_negative(self.records)
        else:
            self.samples = list(self.records)

        num_pos = sum(1 for r in self.records if r.fg_ratio > self.positive_min_ratio)
        num_neg = len(self.records) - num_pos
        num_hard_neg = sum(1 for r in self.records if r.is_hard_negative)
        print(
            f"[FanPatchDataset] split={split}, records={len(self.records)}, samples={len(self.samples)}, "
            f"pos={num_pos}, neg={num_neg}, hard_neg={num_hard_neg}, normalize={normalize_mode}, "
            f"use_hard_negative={self.use_hard_negative}"
        )

    def _load_hard_negative_map_from_meta(self) -> Optional[Dict[str, bool]]:
        if not os.path.exists(self.meta_csv):
            return None

        with open(self.meta_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None

            candidate_keys = ["hard_negative_hint", "is_hard_negative", "hard_negative", "hard_neg"]
            key = next((k for k in candidate_keys if k in reader.fieldnames), None)
            split_key = "split" if "split" in reader.fieldnames else None
            patch_key = "patch_name" if "patch_name" in reader.fieldnames else None
            image_key = "image_path" if "image_path" in reader.fieldnames else None

            if key is None or (patch_key is None and image_key is None):
                return None

            mapping: Dict[str, bool] = {}
            for row in reader:
                if split_key and row.get(split_key) != self.split:
                    continue
                raw_name = row.get(patch_key) or os.path.splitext(os.path.basename(row.get(image_key, "")))[0]
                if not raw_name:
                    continue
                val = str(row.get(key, "")).strip().lower()
                mapping[raw_name] = val in {"1", "true", "yes", "y"}
            return mapping

    def _collect_records(self) -> List[SampleRecord]:
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.npy")))
        records: List[SampleRecord] = []

        hard_neg_map = self._load_hard_negative_map_from_meta()
        if hard_neg_map is not None:
            print(f"[FanPatchDataset] 检测到 meta/samples.csv 中的 hard negative 标记，split={self.split}")

        for img_path in image_paths:
            name = os.path.basename(img_path)
            stem = os.path.splitext(name)[0]
            mask_path = os.path.join(self.mask_dir, name)
            if not os.path.exists(mask_path):
                print(f"[警告] 缺少对应 mask，跳过: {mask_path}")
                continue

            fan_id, row, col = parse_patch_name(name)
            mask = ensure_mask_1hw(np.load(mask_path)).astype(np.float32)
            mask = binarize_mask(mask)
            fg_ratio = float(mask.mean())

            rec = SampleRecord(
                img_path=img_path,
                mask_path=mask_path,
                fan_id=fan_id,
                row=row,
                col=col,
                fg_ratio=fg_ratio,
                is_hard_negative=False,
            )
            if hard_neg_map is not None:
                rec.is_hard_negative = bool(hard_neg_map.get(stem, False))
            records.append(rec)

        if hard_neg_map is None:
            self._mark_hard_negatives(records)

        return records

    def _infer_axis_step(self, coords: List[int], default: int = 128) -> int:
        vals = sorted(set(coords))
        diffs = [b - a for a, b in zip(vals[:-1], vals[1:]) if (b - a) > 0]
        return min(diffs) if diffs else default

    def _mark_hard_negatives(self, records: List[SampleRecord]) -> None:
        by_fan: Dict[str, List[SampleRecord]] = {}
        for r in records:
            by_fan.setdefault(r.fan_id, []).append(r)

        for items in by_fan.values():
            pos_items = [r for r in items if r.fg_ratio > self.positive_min_ratio]
            if not pos_items:
                continue

            row_step = self._infer_axis_step([r.row for r in items], default=128)
            col_step = self._infer_axis_step([r.col for r in items], default=128)
            row_radius = self.hard_negative_radius_steps * row_step
            col_radius = self.hard_negative_radius_steps * col_step

            pos_coords = {(r.row, r.col) for r in pos_items}
            for r in items:
                if r.fg_ratio > self.positive_min_ratio:
                    continue
                for pr, pc in pos_coords:
                    if abs(r.row - pr) <= row_radius and abs(r.col - pc) <= col_radius:
                        r.is_hard_negative = True
                        break

    def _build_train_samples_with_hard_negative(self, records: List[SampleRecord]) -> List[SampleRecord]:
        samples: List[SampleRecord] = []
        for r in records:
            samples.append(r)
            if r.is_hard_negative:
                for _ in range(self.hard_negative_repeat - 1):
                    samples.append(r)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rec = self.samples[index]
        img = np.load(rec.img_path)
        mask = np.load(rec.mask_path)

        img = ensure_image_chw(img, expected_channels=self.expected_channels)
        mask = ensure_mask_1hw(mask)
        mask = binarize_mask(mask)

        fan_stat = None if self.fan_stats is None else self.fan_stats.get(rec.fan_id)
        img = normalize_image(img, mode=self.normalize_mode, fan_stats=fan_stat)

        img_tensor = torch.from_numpy(img.astype(np.float32)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).float()

        if self.return_path:
            return img_tensor, mask_tensor, rec.img_path, rec.mask_path
        return img_tensor, mask_tensor


def check_one_sample(dataset_root: str, split: str = "train", normalize_mode: str = "fan_channel_minmax"):
    ds = FanPatchDataset(
        dataset_root=dataset_root,
        split=split,
        expected_channels=6,
        normalize_mode=normalize_mode,
        return_path=True,
    )
    img_tensor, mask_tensor, img_path, mask_path = ds[0]

    print("=" * 60)
    print("split       :", split)
    print("img_path    :", img_path)
    print("mask_path   :", mask_path)
    print("img shape   :", tuple(img_tensor.shape), img_tensor.dtype)
    print("mask shape  :", tuple(mask_tensor.shape), mask_tensor.dtype)
    print("img min/max :", float(img_tensor.min()), float(img_tensor.max()))
    print("mask unique :", torch.unique(mask_tensor))
    print("=" * 60)


if __name__ == "__main__":
    dataset_root = r"E:\U-net_fan_extract\02_code\04_dataset_multi_ps256_s128"
    check_one_sample(dataset_root, split="train", normalize_mode="fan_channel_minmax")
