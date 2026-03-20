from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch


def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(img.shape)}")
    x = img.detach().cpu().float()
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    x = x.permute(1, 2, 0).contiguous().numpy()
    x = x - x.min()
    x = x / (x.max() + 1e-12)
    return (x * 255.0).astype(np.uint8)


def _to_hw_float(amap: torch.Tensor) -> np.ndarray:
    a = amap.detach().cpu().float()
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"Expected HW or 1HW anomaly map, got shape={tuple(a.shape)}")
    return a.numpy()


def save_anomaly_visualization(
    out_path: str,
    image: torch.Tensor,
    anomaly_map: torch.Tensor,
    gt_mask: Optional[torch.Tensor] = None,
) -> None:
    img = _to_hwc_uint8(image)
    amap = _to_hw_float(anomaly_map)
    amap = amap - float(np.min(amap))
    amap = amap / (float(np.max(amap)) + 1e-12)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img)
        ax1.set_title("image")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(amap, cmap="jet")
        ax2.set_title("anomaly map")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(img)
        ax3.imshow(amap, cmap="jet", alpha=0.5)
        if gt_mask is not None:
            m = gt_mask.detach().cpu()
            if m.ndim == 3:
                m = m[0]
            ax3.contour(m.numpy() > 0.5, colors="white", linewidths=1)
        ax3.set_title("overlay")
        ax3.axis("off")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        from PIL import Image

        heat = (amap * 255.0).astype(np.uint8)
        Image.fromarray(img).save(out_path.replace(".png", "_img.png"))
        Image.fromarray(heat).save(out_path.replace(".png", "_map.png"))


def save_score_histogram(
    out_path: str,
    normal_scores: Sequence[float],
    anomaly_scores: Sequence[float],
    title: str = "image score distribution",
    bins: int = 50,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = np.asarray(normal_scores, dtype=np.float32)
        a = np.asarray(anomaly_scores, dtype=np.float32)
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(1, 1, 1)
        if len(n) > 0:
            ax.hist(n, bins=bins, alpha=0.6, label=f"normal (n={len(n)})")
        if len(a) > 0:
            ax.hist(a, bins=bins, alpha=0.6, label=f"anomaly (n={len(a)})")
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def save_curve_plot(
    out_path: str,
    x: Sequence[float],
    y: Sequence[float],
    x_label: str,
    y_label: str,
    title: str,
    extra_lines: Optional[Sequence[Tuple[Sequence[float], Sequence[float], str]]] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.asarray(x), np.asarray(y), label=title)
        if extra_lines:
            for xx, yy, lab in extra_lines:
                ax.plot(np.asarray(xx), np.asarray(yy), label=lab)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def save_topk_table(
    out_path: str,
    rows: Sequence[dict],
    columns: Sequence[str],
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cell_text = []
        for r in rows:
            cell_text.append([str(r.get(c, "")) for c in columns])

        fig = plt.figure(figsize=(10, max(2, 0.35 * (len(rows) + 1))))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.set_title(title)
        tbl = ax.table(cellText=cell_text, colLabels=list(columns), loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
