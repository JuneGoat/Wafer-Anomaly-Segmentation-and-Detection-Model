from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.anomaly_model import AnomalyModel
from utils.logger import get_logger
from utils.visualization import save_anomaly_visualization

from .metrics import PROConfig, best_f1_threshold, dice_iou, pro_score, safe_auroc, summarize_metrics


@dataclass(frozen=True)
class EvaluatorConfig:
    output_dir: str = "checkpoints"
    save_visualizations: bool = True
    max_visualizations: int = 20
    pro: PROConfig = PROConfig()


class Evaluator:
    def __init__(self, cfg: EvaluatorConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger("evaluator")

    @torch.no_grad()
    def evaluate(
        self,
        model: AnomalyModel,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.to(device)
        model.eval()

        y_true: list[int] = []
        y_score: list[float] = []
        has_masks = False
        maps: list[np.ndarray] = []
        gts: list[np.ndarray] = []

        vis_dir = Path(self.cfg.output_dir) / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        vis_count = 0
        for batch_idx, batch in enumerate(dataloader):
            x = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch.get("mask", None)
            has_mask = batch.get("has_mask", None)

            pred = model.predict(x)
            amap = pred["anomaly_map"].detach().cpu().numpy()
            iscore = pred["image_score"].detach().cpu().numpy()

            y_true.extend([int(v) for v in labels])
            y_score.extend([float(v) for v in iscore])

            if masks is not None and has_mask is not None:
                if torch.is_tensor(has_mask):
                    sel = has_mask.bool()
                else:
                    sel = torch.tensor(has_mask, dtype=torch.bool)
                if sel.any().item():
                    has_masks = True
                    m_t = masks.to(device, non_blocking=True) if torch.is_tensor(masks) else torch.tensor(masks, device=device)
                    gts.append(m_t[sel].detach().cpu().numpy())
                    maps.append(amap[sel.detach().cpu().numpy()])

            if self.cfg.save_visualizations and vis_count < self.cfg.max_visualizations:
                x_cpu = x.detach().cpu()
                mask_cpu = masks.detach().cpu() if torch.is_tensor(masks) else None
                sel_cpu = has_mask.detach().cpu() if torch.is_tensor(has_mask) else None
                for i in range(x_cpu.shape[0]):
                    if vis_count >= self.cfg.max_visualizations:
                        break
                    gt_vis = None
                    if mask_cpu is not None and sel_cpu is not None and bool(sel_cpu[i].item()):
                        gt_vis = mask_cpu[i]
                    save_anomaly_visualization(
                        out_path=str(vis_dir / f"sample_{batch_idx:04d}_{i:02d}.png"),
                        image=x_cpu[i],
                        anomaly_map=torch.from_numpy(amap[i]),
                        gt_mask=gt_vis,
                    )
                    vis_count += 1

        y_true_np = np.asarray(y_true, dtype=np.int32)
        y_score_np = np.asarray(y_score, dtype=np.float32)

        thr_img, _, _ = best_f1_threshold(y_true_np, y_score_np) if y_true_np.min() != y_true_np.max() else (float(np.median(y_score_np)), 0.0, 0.0)
        model.set_thresholds(image_score_threshold=thr_img, map_threshold=None)

        mask_metrics: Dict[str, float] = {}
        if has_masks and maps and gts:
            amap_all = np.concatenate(maps, axis=0)
            gt_all = np.concatenate(gts, axis=0)
            amap_flat = amap_all.reshape(amap_all.shape[0], -1)
            dice_list = []
            iou_list = []
            pro_list = []
            best_thr = None
            best_dice = -1.0
            for thr in np.quantile(amap_flat, np.linspace(0.8, 0.999, 20)):
                dices = []
                ious = []
                for i in range(amap_all.shape[0]):
                    pred_mask = (amap_all[i, 0] >= thr).astype(np.uint8)
                    gt_mask = (gt_all[i, 0] > 0).astype(np.uint8)
                    d, j = dice_iou(pred_mask, gt_mask)
                    dices.append(d)
                    ious.append(j)
                md = float(np.mean(dices))
                if md > best_dice:
                    best_dice = md
                    best_thr = float(thr)

            if best_thr is None:
                best_thr = float(np.quantile(amap_flat, 0.99))
            model.set_thresholds(image_score_threshold=thr_img, map_threshold=best_thr)

            for i in range(amap_all.shape[0]):
                pred_mask = (amap_all[i, 0] >= best_thr).astype(np.uint8)
                gt_mask = (gt_all[i, 0] > 0).astype(np.uint8)
                d, j = dice_iou(pred_mask, gt_mask)
                dice_list.append(d)
                iou_list.append(j)
                pro_list.append(pro_score(amap_all[i, 0], gt_all[i, 0], cfg=self.cfg.pro))

            mask_metrics = {
                "pixel_dice": float(np.nanmean(np.asarray(dice_list, dtype=np.float32))),
                "pixel_iou": float(np.nanmean(np.asarray(iou_list, dtype=np.float32))),
                "pixel_pro": float(np.nanmean(np.asarray(pro_list, dtype=np.float32))),
                "pixel_best_threshold": float(best_thr),
            }

        out = summarize_metrics(y_true_np, y_score_np, mask_metrics=mask_metrics)
        out["image_threshold"] = float(thr_img)
        out["image_auroc"] = float(safe_auroc(y_true_np, y_score_np))

        out_path = Path(self.cfg.output_dir) / "metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        self.logger.info(f"Evaluation complete. metrics_path={out_path}")
        return out
