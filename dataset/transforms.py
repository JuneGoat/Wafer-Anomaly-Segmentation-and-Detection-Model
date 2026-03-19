from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DataTransformConfig:
    image_size: Tuple[int, int] = (256, 256)
    mean: Tuple[float, ...] = (0.5,)
    std: Tuple[float, ...] = (0.5,)
    augment: bool = True
    random_rotate90: bool = True
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5


def _ensure_chw_float(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW or HW tensor, got shape={tuple(x.shape)}")
    if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        x = x.float() / 255.0
    else:
        x = x.float()
    return x


def _resize_chw(
    x: torch.Tensor,
    size_hw: Tuple[int, int],
    mode: str,
) -> torch.Tensor:
    x4 = x.unsqueeze(0)
    y4 = F.interpolate(x4, size=size_hw, mode=mode, align_corners=False if mode in ("bilinear", "bicubic") else None)
    return y4.squeeze(0)


def _normalize(x: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    if mean_t.shape[0] == 1 and x.shape[0] != 1:
        mean_t = mean_t.expand(x.shape[0], 1, 1)
        std_t = std_t.expand(x.shape[0], 1, 1)
    if mean_t.shape[0] != x.shape[0]:
        raise ValueError(f"Normalization channels mismatch: image C={x.shape[0]} vs mean/std C={mean_t.shape[0]}")
    return (x - mean_t) / (std_t + 1e-12)


def build_transforms(
    cfg: DataTransformConfig,
    mode: str,
) -> Callable[[torch.Tensor, Optional[torch.Tensor]], tuple[torch.Tensor, Optional[torch.Tensor]]]:
    if mode not in ("train", "test"):
        raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")

    def _apply(image: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        image = _ensure_chw_float(image)
        if mask is not None:
            mask = _ensure_chw_float(mask)
            if mask.shape[0] != 1:
                mask = mask[:1]

        if mode == "train" and cfg.augment:
            if cfg.random_rotate90:
                k = int(torch.randint(low=0, high=4, size=(1,)).item())
                if k:
                    image = torch.rot90(image, k, dims=(1, 2))
                    if mask is not None:
                        mask = torch.rot90(mask, k, dims=(1, 2))
            if cfg.hflip_prob > 0 and torch.rand(()) < cfg.hflip_prob:
                image = torch.flip(image, dims=(2,))
                if mask is not None:
                    mask = torch.flip(mask, dims=(2,))
            if cfg.vflip_prob > 0 and torch.rand(()) < cfg.vflip_prob:
                image = torch.flip(image, dims=(1,))
                if mask is not None:
                    mask = torch.flip(mask, dims=(1,))

        image = _resize_chw(image, cfg.image_size, mode="bilinear")
        if mask is not None:
            mask = _resize_chw(mask, cfg.image_size, mode="nearest")

        image = _normalize(image, cfg.mean, cfg.std)
        return image, mask

    return _apply
