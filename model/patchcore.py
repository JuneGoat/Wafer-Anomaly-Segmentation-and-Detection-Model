from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .backbone import Backbone, upsample_to


@dataclass(frozen=True)
class PatchCoreConfig:
    max_features: int = 200_000
    knn_k: int = 1
    image_score_topk: float = 0.01
    random_seed: int = 0


def _concat_multiscale(features: List[torch.Tensor]) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("Empty feature list")
    target_hw = features[0].shape[-2:]
    aligned = [upsample_to(f, target_hw) for f in features]
    return torch.cat(aligned, dim=1)


def _flatten_patches(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    return feat.permute(0, 2, 3, 1).reshape(b * h * w, c)


class PatchCore:
    def __init__(self, cfg: PatchCoreConfig) -> None:
        self.cfg = cfg
        self._memory: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self.feature_hw: Optional[Tuple[int, int]] = None
        self.embed_dim: Optional[int] = None
        self.score_mean: float = 0.0
        self.score_std: float = 1.0

    @property
    def is_fitted(self) -> bool:
        return self._nn is not None and self._memory is not None

    @torch.no_grad()
    def build_memory(
        self,
        backbone: Backbone,
        dataloader: Iterable[dict],
        device: torch.device,
    ) -> None:
        backbone.eval()
        rng = np.random.default_rng(self.cfg.random_seed)

        feats_np: List[np.ndarray] = []
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            feats = backbone(images)
            feat = _concat_multiscale(feats)
            self.feature_hw = (feat.shape[-2], feat.shape[-1])
            flat = _flatten_patches(feat)
            flat = F.normalize(flat, dim=1)
            feats_np.append(flat.detach().cpu().numpy().astype(np.float32))

        all_feats = np.concatenate(feats_np, axis=0)
        self.embed_dim = all_feats.shape[1]

        if all_feats.shape[0] > self.cfg.max_features:
            idx = rng.choice(all_feats.shape[0], size=self.cfg.max_features, replace=False)
            all_feats = all_feats[idx]

        self._memory = all_feats
        self._nn = NearestNeighbors(n_neighbors=self.cfg.knn_k, algorithm="auto", metric="euclidean")
        self._nn.fit(self._memory)

        dists, _ = self._nn.kneighbors(self._memory, n_neighbors=1, return_distance=True)
        self.score_mean = float(dists.mean())
        self.score_std = float(dists.std() + 1e-12)

    @torch.no_grad()
    def score(
        self,
        backbone: Backbone,
        images: torch.Tensor,
        out_size_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("PatchCore memory is not built. Call build_memory() first.")
        assert self._nn is not None
        assert self.feature_hw is not None

        feats = backbone(images)
        feat = _concat_multiscale(feats)
        flat = _flatten_patches(feat)
        flat = F.normalize(flat, dim=1)
        q = flat.detach().cpu().numpy().astype(np.float32)

        dists, _ = self._nn.kneighbors(q, n_neighbors=self.cfg.knn_k, return_distance=True)
        if dists.ndim == 2 and dists.shape[1] > 1:
            d = dists.mean(axis=1)
        else:
            d = dists.reshape(-1)

        d = (d - self.score_mean) / self.score_std
        b = images.shape[0]
        h, w = self.feature_hw
        score_map = torch.from_numpy(d).to(images.device).view(b, 1, h, w)

        if out_size_hw is not None and score_map.shape[-2:] != out_size_hw:
            score_map = F.interpolate(score_map, size=out_size_hw, mode="bilinear", align_corners=False)

        topk = max(1, int(score_map.numel() / b * self.cfg.image_score_topk))
        image_score = torch.topk(score_map.view(b, -1), k=topk, dim=1).values.mean(dim=1)
        return score_map, image_score
