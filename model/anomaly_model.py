from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import ConvAutoencoder, ConvAutoencoderConfig
from .backbone import Backbone, BackboneConfig
from .patchcore import PatchCore, PatchCoreConfig


@dataclass(frozen=True)
class AnomalyModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    autoencoder: ConvAutoencoderConfig = field(default_factory=ConvAutoencoderConfig)
    use_patchcore: bool = True
    use_autoencoder: bool = True
    weight_patchcore: float = 1.0
    weight_recon: float = 1.0


class AnomalyModel(nn.Module):
    def __init__(self, cfg: AnomalyModelConfig, in_channels: int) -> None:
        super().__init__()
        self.cfg = cfg

        self.backbone = Backbone(cfg.backbone, in_channels=in_channels)
        self.patchcore = PatchCore(cfg.patchcore)
        self.autoencoder: Optional[ConvAutoencoder] = ConvAutoencoder(
            ConvAutoencoderConfig(
                in_channels=in_channels,
                out_channels=in_channels,
                base_channels=cfg.autoencoder.base_channels,
                num_down=cfg.autoencoder.num_down,
            )
        )

        if not cfg.use_autoencoder:
            self.autoencoder = None

        self.recon_mean: float = 0.0
        self.recon_std: float = 1.0
        self.recon_score_mean: float = 0.0
        self.recon_score_std: float = 1.0

        self.score_threshold: Optional[float] = None
        self.map_threshold: Optional[float] = None

        for p in self.backbone.parameters():
            p.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def build_feature_memory(self, train_loader: Any, device: torch.device) -> None:
        if not self.cfg.use_patchcore:
            return
        self.backbone.to(device)
        self.patchcore.build_memory(self.backbone, train_loader, device=device)

    @torch.no_grad()
    def calibrate_reconstruction(self, train_loader: Any, device: torch.device, max_batches: int = 100) -> None:
        if self.autoencoder is None:
            return
        self.autoencoder.to(device)
        self.autoencoder.eval()

        maps: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            x = batch["image"].to(device, non_blocking=True)
            err_map = self.autoencoder.reconstruction_error_map(x)
            maps.append(err_map.detach().cpu().numpy().astype(np.float32).reshape(-1))
            scores.append(err_map.view(x.shape[0], -1).mean(dim=1).detach().cpu().numpy().astype(np.float32))

        if not maps:
            return
        m = np.concatenate(maps, axis=0)
        s = np.concatenate(scores, axis=0)
        self.recon_mean = float(m.mean())
        self.recon_std = float(m.std() + 1e-12)
        self.recon_score_mean = float(s.mean())
        self.recon_score_std = float(s.std() + 1e-12)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[-2:]

        patch_map = torch.zeros((x.shape[0], 1, h, w), device=x.device)
        patch_score = torch.zeros((x.shape[0],), device=x.device)
        if self.cfg.use_patchcore:
            patch_map, patch_score = self.patchcore.score(self.backbone, x, out_size_hw=(h, w))

        recon = torch.zeros_like(x)
        recon_map = torch.zeros((x.shape[0], 1, h, w), device=x.device)
        recon_score = torch.zeros((x.shape[0],), device=x.device)
        if self.autoencoder is not None:
            recon = self.autoencoder(x)
            recon_map = (recon - x).pow(2)
            if recon_map.shape[1] > 1:
                recon_map = recon_map.mean(dim=1, keepdim=True)
            recon_score = recon_map.view(x.shape[0], -1).mean(dim=1)

        patch_map_n = patch_map
        patch_score_n = patch_score
        if self.cfg.use_patchcore:
            patch_map_n = patch_map
            patch_score_n = patch_score

        recon_map_n = recon_map
        recon_score_n = recon_score
        if self.autoencoder is not None:
            recon_map_n = (recon_map - self.recon_mean) / self.recon_std
            recon_score_n = (recon_score - self.recon_score_mean) / self.recon_score_std

        w_p = float(self.cfg.weight_patchcore) if self.cfg.use_patchcore else 0.0
        w_r = float(self.cfg.weight_recon) if self.autoencoder is not None else 0.0
        w_sum = max(1e-12, w_p + w_r)

        anomaly_map = (w_p * patch_map_n + w_r * recon_map_n) / w_sum
        image_score = (w_p * patch_score_n + w_r * recon_score_n) / w_sum

        out: Dict[str, torch.Tensor] = {
            "anomaly_map": anomaly_map,
            "image_score": image_score,
            "patch_map": patch_map,
            "patch_score": patch_score,
            "recon": recon,
            "recon_map": recon_map,
            "recon_score": recon_score,
        }

        if self.map_threshold is not None:
            out["pred_mask"] = (anomaly_map >= self.map_threshold).float()
        else:
            out["pred_mask"] = torch.zeros_like(anomaly_map)

        return out

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.predict(x)

    def set_thresholds(self, image_score_threshold: Optional[float], map_threshold: Optional[float]) -> None:
        self.score_threshold = image_score_threshold
        self.map_threshold = map_threshold
