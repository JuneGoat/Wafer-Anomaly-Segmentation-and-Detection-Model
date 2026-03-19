from __future__ import annotations

from typing import Optional

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
