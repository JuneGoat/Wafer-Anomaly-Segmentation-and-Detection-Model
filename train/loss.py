from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReconstructionLossConfig:
    reduction: str = "mean"


class ReconstructionLoss:
    def __init__(self, cfg: ReconstructionLossConfig) -> None:
        self.cfg = cfg

    def __call__(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target, reduction=self.cfg.reduction)
