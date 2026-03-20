from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(np.int32), y_score.astype(np.float32))
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))
    thr = float(thresholds[min(best_idx, len(thresholds) - 1)]) if len(thresholds) else 0.5
    return thr, float(precision[best_idx]), float(recall[best_idx])


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = y_true.astype(np.int32)
    if y_true.min() == y_true.max():
        fpr = np.asarray([0.0, 1.0], dtype=np.float32)
        tpr = np.asarray([0.0, 1.0], dtype=np.float32)
        thr = np.asarray([np.inf, -np.inf], dtype=np.float32)
        return fpr, tpr, thr
    fpr, tpr, thr = roc_curve(y_true, y_score.astype(np.float32))
    return fpr.astype(np.float32), tpr.astype(np.float32), thr.astype(np.float32)


def pr_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true.astype(np.int32), y_score.astype(np.float32))
    thr = thresholds.astype(np.float32) if len(thresholds) else np.asarray([], dtype=np.float32)
    return precision.astype(np.float32), recall.astype(np.float32), thr


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    iou = inter / (union + 1e-12)
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-12)
    return dice, iou


def _connected_components(bin_mask: np.ndarray) -> list[np.ndarray]:
    h, w = bin_mask.shape
    visited = np.zeros_like(bin_mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for y in range(h):
        for x in range(w):
            if not bin_mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and bin_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(pts)

    out = []
    for pts in comps:
        m = np.zeros_like(bin_mask, dtype=np.uint8)
        ys, xs = zip(*pts)
        m[list(ys), list(xs)] = 1
        out.append(m)
    return out


@dataclass(frozen=True)
class PROConfig:
    num_thresholds: int = 50
    fpr_limit: float = 0.3


def pro_score(anomaly_map: np.ndarray, gt_mask: np.ndarray, cfg: Optional[PROConfig] = None) -> float:
    cfg = cfg or PROConfig()
    amap = anomaly_map.astype(np.float32)
    gt = (gt_mask > 0).astype(np.uint8)
    if gt.sum() == 0:
        return float("nan")

    thresholds = np.linspace(float(amap.min()), float(amap.max()), cfg.num_thresholds)
    bg = (gt == 0).astype(np.uint8)
    comps = _connected_components(gt)
    if not comps:
        return float("nan")

    pros = []
    fprs = []
    bg_total = float(bg.sum() + 1e-12)
    for thr in thresholds:
        pred = (amap >= thr).astype(np.uint8)
        fpr = float((pred & bg).sum()) / bg_total
        if fpr > cfg.fpr_limit:
            continue
        overlaps = []
        for comp in comps:
            comp_area = float(comp.sum() + 1e-12)
            overlaps.append(float((pred & comp).sum()) / comp_area)
        pros.append(float(np.mean(overlaps)))
        fprs.append(fpr)

    if len(pros) < 2:
        return float("nan")
    order = np.argsort(np.array(fprs))
    fprs_s = np.array(fprs)[order]
    pros_s = np.array(pros)[order]
    auc = float(np.trapz(pros_s, fprs_s) / (cfg.fpr_limit + 1e-12))
    return auc


def summarize_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    thr, p, r = best_f1_threshold(y_true, y_score)
    out = {
        "image_auroc": safe_auroc(y_true, y_score),
        "image_ap": average_precision(y_true, y_score),
        "image_best_f1_threshold": float(thr),
        "image_precision_at_best_f1": float(p),
        "image_recall_at_best_f1": float(r),
    }
    if mask_metrics:
        out.update(mask_metrics)
    return out
