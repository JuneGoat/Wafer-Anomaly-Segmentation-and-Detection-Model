from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from dataset.transforms import DataTransformConfig
from dataset.wafer_dataset import WaferDataset
from eval.evaluator import Evaluator, EvaluatorConfig
from model.anomaly_model import AnomalyModel, AnomalyModelConfig
from model.autoencoder import ConvAutoencoderConfig
from model.backbone import BackboneConfig
from model.patchcore import PatchCoreConfig
from train.trainer import Trainer, TrainerConfig
from utils.config import load_config, set_global_seed
from utils.logger import get_logger


def _to_dataclass(cfg: Dict[str, Any], cls: Any) -> Any:
    fields = {k: v for k, v in cfg.items() if k in getattr(cls, "__dataclass_fields__", {})}
    return cls(**fields)


def build_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]
    tcfg = DataTransformConfig(
        image_size=tuple(dcfg["image_size"]),
        mean=tuple(dcfg["mean"]),
        std=tuple(dcfg["std"]),
        augment=True,
    )

    train_ds = WaferDataset(
        data_root=str(dcfg["root"]),
        split="train",
        input_type=str(dcfg["input_type"]),
        channels=int(dcfg.get("channels", 3)),
        transform_cfg=tcfg,
    )
    test_ds = WaferDataset(
        data_root=str(dcfg["root"]),
        split="test",
        input_type=str(dcfg["input_type"]),
        channels=int(dcfg.get("channels", 3)),
        transform_cfg=DataTransformConfig(
            image_size=tuple(dcfg["image_size"]),
            mean=tuple(dcfg["mean"]),
            std=tuple(dcfg["std"]),
            augment=False,
        ),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(dcfg["batch_size"]),
        shuffle=True,
        num_workers=int(dcfg.get("num_workers", 2)),
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(dcfg["batch_size"]),
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 2)),
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def build_model(cfg: Dict[str, Any]) -> AnomalyModel:
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    in_channels = int(dcfg.get("channels", 3))

    backbone_cfg = _to_dataclass(mcfg["backbone"], BackboneConfig)
    patch_cfg = _to_dataclass(mcfg["patchcore"], PatchCoreConfig)
    ae_cfg = _to_dataclass(mcfg["autoencoder"], ConvAutoencoderConfig)

    model_cfg = AnomalyModelConfig(
        backbone=backbone_cfg,
        patchcore=patch_cfg,
        autoencoder=ConvAutoencoderConfig(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=int(ae_cfg.base_channels),
            num_down=int(ae_cfg.num_down),
        ),
        use_patchcore=bool(mcfg.get("use_patchcore", True)),
        use_autoencoder=bool(mcfg.get("use_autoencoder", True)),
        weight_patchcore=float(mcfg.get("weight_patchcore", 1.0)),
        weight_recon=float(mcfg.get("weight_recon", 1.0)),
    )

    return AnomalyModel(model_cfg, in_channels=in_channels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("main")

    seed = int(cfg.get("seed", 0))
    set_global_seed(seed)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg)

    if args.dry_run:
        batch = next(iter(train_loader))
        x = batch["image"].to(device)
        model.to(device)
        with torch.no_grad():
            out = model.predict(x)
        logger.info(f"Dry run OK. anomaly_map={tuple(out['anomaly_map'].shape)} image_score={tuple(out['image_score'].shape)}")
        return

    trainer_cfg = _to_dataclass(cfg.get("train", {}), TrainerConfig)
    trainer = Trainer(trainer_cfg)

    run_config: Dict[str, Any] = cfg
    trainer.train(model, train_loader=train_loader, device=device, resume_from=args.resume, run_config=run_config)

    eval_cfg = cfg.get("eval", {})
    evaluator_cfg = EvaluatorConfig(
        output_dir=str(eval_cfg.get("output_dir", trainer_cfg.checkpoint_dir)),
        save_visualizations=bool(eval_cfg.get("save_visualizations", True)),
        max_visualizations=int(eval_cfg.get("max_visualizations", 20)),
        save_analysis=bool(eval_cfg.get("save_analysis", True)),
        topk=int(eval_cfg.get("topk", 50)),
        pro=_to_dataclass(eval_cfg.get("pro", {}), type(EvaluatorConfig().pro)),
    )
    evaluator = Evaluator(evaluator_cfg)
    metrics = evaluator.evaluate(model, dataloader=test_loader, device=device)

    logger.info("Metrics:")
    for k in sorted(metrics.keys()):
        logger.info(f"{k}: {metrics[k]}")


if __name__ == "__main__":
    main()
