from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.anomaly_model import AnomalyModel
from utils.logger import get_logger

from .loss import ReconstructionLoss, ReconstructionLossConfig


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    mixed_precision: bool = True
    grad_clip_norm: Optional[float] = None
    log_every: int = 20
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "latest.pt"
    save_every_epochs: int = 1
    recon_loss: ReconstructionLossConfig = field(default_factory=ReconstructionLossConfig)


def save_checkpoint(
    path: str,
    model: AnomalyModel,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    config: Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "config": config,
        "recon_mean": float(model.recon_mean),
        "recon_std": float(model.recon_std),
        "recon_score_mean": float(model.recon_score_mean),
        "recon_score_std": float(model.recon_score_std),
        "patchcore_mean": float(model.patchcore.score_mean),
        "patchcore_std": float(model.patchcore.score_std),
        "patchcore_memory": model.patchcore._memory,
        "patchcore_feature_hw": model.patchcore.feature_hw,
        "patchcore_embed_dim": model.patchcore.embed_dim,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: AnomalyModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> int:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    model.recon_mean = float(ckpt.get("recon_mean", 0.0))
    model.recon_std = float(ckpt.get("recon_std", 1.0))
    model.recon_score_mean = float(ckpt.get("recon_score_mean", 0.0))
    model.recon_score_std = float(ckpt.get("recon_score_std", 1.0))

    model.patchcore.score_mean = float(ckpt.get("patchcore_mean", 0.0))
    model.patchcore.score_std = float(ckpt.get("patchcore_std", 1.0))
    model.patchcore._memory = ckpt.get("patchcore_memory", None)
    model.patchcore.feature_hw = tuple(ckpt.get("patchcore_feature_hw")) if ckpt.get("patchcore_feature_hw") else None
    model.patchcore.embed_dim = ckpt.get("patchcore_embed_dim", None)
    if model.patchcore._memory is not None:
        from sklearn.neighbors import NearestNeighbors

        model.patchcore._nn = NearestNeighbors(n_neighbors=model.patchcore.cfg.knn_k, algorithm="auto", metric="euclidean")
        model.patchcore._nn.fit(model.patchcore._memory)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


class Trainer:
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger("trainer")

    def train(
        self,
        model: AnomalyModel,
        train_loader: DataLoader,
        device: torch.device,
        resume_from: Optional[str] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        run_config = run_config or {}
        model.to(device)

        optim_params = []
        if model.autoencoder is not None:
            optim_params += list(model.autoencoder.parameters())
        optimizer = torch.optim.AdamW(optim_params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay) if optim_params else None

        start_epoch = 0
        ckpt_path = str(Path(self.cfg.checkpoint_dir) / self.cfg.checkpoint_name)
        if resume_from is not None and Path(resume_from).exists():
            start_epoch = load_checkpoint(resume_from, model=model, optimizer=optimizer, map_location=device)
            self.logger.info(f"Resumed from {resume_from} at epoch {start_epoch}")
        elif Path(ckpt_path).exists():
            start_epoch = load_checkpoint(ckpt_path, model=model, optimizer=optimizer, map_location=device)
            self.logger.info(f"Resumed from {ckpt_path} at epoch {start_epoch}")

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.mixed_precision and device.type == "cuda")
            autocast = torch.autocast
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision and device.type == "cuda")
            autocast = torch.cuda.amp.autocast
        loss_fn = ReconstructionLoss(self.cfg.recon_loss)

        if model.autoencoder is not None and optimizer is not None:
            model.autoencoder.train()
            global_step = 0
            for epoch in range(start_epoch, self.cfg.epochs):
                epoch_start = time.time()
                running = 0.0
                n_batches = 0
                for batch_idx, batch in enumerate(train_loader):
                    x = batch["image"].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    if autocast is torch.autocast:
                        ctx = autocast(device_type=device.type, enabled=scaler.is_enabled())
                    else:
                        ctx = autocast(enabled=scaler.is_enabled())
                    with ctx:
                        recon = model.autoencoder(x)
                        loss = loss_fn(recon, x)
                    scaler.scale(loss).backward()
                    if self.cfg.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.autoencoder.parameters(), self.cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    running += float(loss.detach().item())
                    n_batches += 1
                    global_step += 1
                    if self.cfg.log_every > 0 and (batch_idx + 1) % self.cfg.log_every == 0:
                        self.logger.info(
                            f"epoch={epoch+1}/{self.cfg.epochs} step={batch_idx+1}/{len(train_loader)} "
                            f"loss={running/max(1,n_batches):.6f}"
                        )

                elapsed = time.time() - epoch_start
                self.logger.info(
                    f"epoch={epoch+1}/{self.cfg.epochs} recon_loss={running/max(1,n_batches):.6f} time={elapsed:.1f}s"
                )

                if (epoch + 1) % self.cfg.save_every_epochs == 0:
                    save_checkpoint(
                        ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        config=run_config,
                    )

            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=self.cfg.epochs, config=run_config)

        if model.cfg.use_patchcore:
            model.build_feature_memory(train_loader, device=device)
            save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=self.cfg.epochs, config=run_config)

        model.calibrate_reconstruction(train_loader, device=device)
        save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=self.cfg.epochs, config=run_config)

        meta_path = str(Path(self.cfg.checkpoint_dir) / "run_config.json")
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False)
