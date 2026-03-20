from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import DataTransformConfig, build_transforms


@dataclass(frozen=True)
class WaferSample:
    image_path: str
    label: int
    mask_path: Optional[str] = None


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _is_array_file(p: Path) -> bool:
    return p.suffix.lower() in {".npy", ".npz", ".csv"}


def _read_image_as_chw_float(path: str, channels: int) -> torch.Tensor:
    img = Image.open(path)
    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    arr = torch.from_numpy(np.array(img))
    if arr.ndim == 2:
        arr = arr.unsqueeze(-1)
    chw = arr.permute(2, 0, 1).contiguous()
    return chw


def _detect_csv_usecols(path: str) -> Optional[Sequence[int]]:
    try:
        with open(path, "r", newline="") as f:
            first = f.readline()
        header = [c.strip() for c in first.split(",")]
        if not header:
            return None
        if header[0] == "":
            return list(range(1, len(header)))
        return None
    except Exception:
        return None


def _read_wafer_map_as_chw_float(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(path)
    elif p.suffix.lower() == ".npz":
        data = np.load(path)
        if len(data.files) == 0:
            raise ValueError(f"Empty npz file: {path}")
        arr = data[data.files[0]]
    elif p.suffix.lower() == ".csv":
        usecols = _detect_csv_usecols(path)
        arr = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=usecols, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported wafer map file type: {path}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D wafer map, got shape={arr.shape} from {path}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.from_numpy(arr).float().unsqueeze(0)
    return t


def _discover_samples(data_root: Path, split: str) -> List[WaferSample]:
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    samples: List[WaferSample] = []

    split_dir = data_root / split
    if split_dir.exists():
        normal_dir = split_dir / "normal"
        anomaly_dir = split_dir / "anomaly"
        mask_dir = split_dir / "masks"

        if normal_dir.exists():
            for p in sorted(normal_dir.rglob("*")):
                if p.is_file() and (_is_image_file(p) or _is_array_file(p)):
                    samples.append(WaferSample(str(p), 0, None))
        if split == "test" and anomaly_dir.exists():
            for p in sorted(anomaly_dir.rglob("*")):
                if p.is_file() and (_is_image_file(p) or _is_array_file(p)):
                    mask_path = None
                    if mask_dir.exists():
                        candidate = mask_dir / p.name
                        if candidate.exists():
                            mask_path = str(candidate)
                    samples.append(WaferSample(str(p), 1, mask_path))

    if samples:
        return samples

    for p in sorted(data_root.rglob("*")):
        if not p.is_file():
            continue
        if _is_image_file(p) or _is_array_file(p):
            samples.append(WaferSample(str(p), 0, None))
    return samples


def _load_list_file(path: str) -> List[WaferSample]:
    samples: List[WaferSample] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"List file must have columns {sorted(required)}; got {reader.fieldnames}")
        for row in reader:
            p = row["path"]
            label = int(row["label"])
            mask_path = row.get("mask") or None
            samples.append(WaferSample(p, label, mask_path))
    return samples


class WaferDataset(Dataset[Dict[str, Any]]):
    """
    Flexible wafer inspection dataset supporting:
      - Image inputs (PNG/JPG/...)
      - Wafer maps / tabular arrays (NPY/NPZ/CSV)

    Output item:
      {
        "image": FloatTensor[C,H,W],
        "label": int (0 normal, 1 anomaly),
        "mask": optional FloatTensor[1,H,W] (0/1)
      }
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        input_type: str,
        transform_cfg: Optional[DataTransformConfig] = None,
        list_file: Optional[str] = None,
        channels: Optional[int] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.input_type = input_type
        if self.input_type not in ("image", "wafer_map"):
            raise ValueError(f"input_type must be 'image' or 'wafer_map', got {self.input_type!r}")

        self.channels = channels if channels is not None else (3 if self.input_type == "image" else 1)
        self.transform_cfg = transform_cfg or DataTransformConfig()
        self.transform = build_transforms(self.transform_cfg, mode="train" if split == "train" else "test")

        if list_file is not None:
            self.samples = _load_list_file(list_file)
        else:
            self.samples = _discover_samples(self.data_root, split=split)

        if split == "train":
            self.samples = [s for s in self.samples if s.label == 0]

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found. data_root={data_root}, split={split}, list_file={list_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, path: str) -> torch.Tensor:
        if self.input_type == "image":
            return _read_image_as_chw_float(path, channels=self.channels)
        return _read_wafer_map_as_chw_float(path)

    def _read_mask(self, path: Optional[str]) -> Optional[torch.Tensor]:
        if path is None:
            return None
        if not os.path.exists(path):
            return None
        return _read_image_as_chw_float(path, channels=1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image = self._read_image(sample.image_path)
        mask = self._read_mask(sample.mask_path)
        has_mask = mask is not None
        image, mask = self.transform(image, mask)

        out: Dict[str, Any] = {
            "image": image,
            "label": int(sample.label),
            "path": sample.image_path,
        }
        if mask is None:
            out["mask"] = torch.zeros((1, image.shape[-2], image.shape[-1]), dtype=image.dtype)
        else:
            out["mask"] = (mask > 0.5).float()
        out["has_mask"] = bool(has_mask)
        return out
