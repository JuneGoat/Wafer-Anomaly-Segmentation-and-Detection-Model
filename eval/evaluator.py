from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.anomaly_model import AnomalyModel
from utils.logger import get_logger
from utils.visualization import save_anomaly_visualization, save_curve_plot, save_score_histogram, save_topk_table

from .metrics import PROConfig, best_f1_threshold, dice_iou, pr_points, pro_score, roc_points, safe_auroc, summarize_metrics


@dataclass(frozen=True)
class EvaluatorConfig:
    output_dir: str = "checkpoints"
    save_visualizations: bool = True
    max_visualizations: int = 20
    save_analysis: bool = True
    topk: int = 50
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
        patch_score_list: list[float] = []
        recon_score_list: list[float] = []
        paths: list[str] = []
        has_masks = False
        maps: list[np.ndarray] = []
        gts: list[np.ndarray] = []

        analysis_dir = Path(self.cfg.output_dir) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = analysis_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        vis_count = 0
        for batch_idx, batch in enumerate(dataloader):
            x = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            batch_paths = batch.get("path", None)
            masks = batch.get("mask", None)
            has_mask = batch.get("has_mask", None)

            pred = model.predict(x)
            amap = pred["anomaly_map"].detach().cpu().numpy()
            iscore = pred["image_score"].detach().cpu().numpy()
            pscore = pred.get("patch_score", torch.zeros_like(pred["image_score"])).detach().cpu().numpy()
            rscore = pred.get("recon_score", torch.zeros_like(pred["image_score"])).detach().cpu().numpy()

            y_true.extend([int(v) for v in labels])
            y_score.extend([float(v) for v in iscore])
            patch_score_list.extend([float(v) for v in pscore])
            recon_score_list.extend([float(v) for v in rscore])
            if batch_paths is None:
                paths.extend([f"idx_{len(paths) + i}" for i in range(len(labels))])
            else:
                if isinstance(batch_paths, (list, tuple)):
                    paths.extend([str(p) for p in batch_paths])
                else:
                    paths.extend([str(p) for p in list(batch_paths)])

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

        if self.cfg.save_analysis:
            rows = []
            for i in range(len(y_true)):
                rows.append(
                    {
                        "path": paths[i] if i < len(paths) else f"idx_{i}",
                        "label": int(y_true[i]),
                        "image_score": float(y_score[i]),
                        "patch_score": float(patch_score_list[i]) if i < len(patch_score_list) else 0.0,
                        "recon_score": float(recon_score_list[i]) if i < len(recon_score_list) else 0.0,
                    }
                )

            csv_path = analysis_dir / "per_sample_scores.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["path", "label", "image_score", "patch_score", "recon_score"])
                w.writeheader()
                w.writerows(rows)

            order = np.argsort(-y_score_np)
            topk = int(min(self.cfg.topk, len(order)))
            top_rows = []
            for j in range(topk):
                i = int(order[j])
                top_rows.append(
                    {
                        "rank": j + 1,
                        "label": int(y_true_np[i]),
                        "image_score": float(y_score_np[i]),
                        "path": paths[i] if i < len(paths) else f"idx_{i}",
                    }
                )
            save_topk_table(
                out_path=str(analysis_dir / "topk_by_image_score.png"),
                rows=top_rows,
                columns=["rank", "label", "image_score", "path"],
                title=f"Top-{topk} samples by image_score",
            )

            normal_scores = [float(y_score[i]) for i in range(len(y_true)) if int(y_true[i]) == 0]
            anomaly_scores = [float(y_score[i]) for i in range(len(y_true)) if int(y_true[i]) == 1]
            save_score_histogram(
                out_path=str(analysis_dir / "image_score_hist.png"),
                normal_scores=normal_scores,
                anomaly_scores=anomaly_scores,
                title="image_score distribution",
            )

            fpr, tpr, _ = roc_points(y_true_np, y_score_np)
            save_curve_plot(
                out_path=str(analysis_dir / "roc_curve.png"),
                x=fpr.tolist(),
                y=tpr.tolist(),
                x_label="FPR",
                y_label="TPR",
                title=f"ROC (AUROC={out['image_auroc']:.4f})",
                extra_lines=[([0.0, 1.0], [0.0, 1.0], "random")],
            )

            precision, recall, _ = pr_points(y_true_np, y_score_np)
            ap = float(out.get("image_ap", float("nan")))
            save_curve_plot(
                out_path=str(analysis_dir / "pr_curve.png"),
                x=recall.tolist(),
                y=precision.tolist(),
                x_label="Recall",
                y_label="Precision",
                title=f"PR (AP={ap:.4f})",
            )

        self.logger.info(f"Evaluation complete. metrics_path={out_path}")
        return out
