from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@dataclass(frozen=True)
class BackboneConfig:
    name: str = "resnet18"
    pretrained: bool = False
    out_layers: Sequence[str] = ("layer2", "layer3", "layer4")


class Backbone(nn.Module):
    def __init__(self, cfg: BackboneConfig, in_channels: int = 3) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        if cfg.name.lower().startswith("resnet"):
            self.kind = "resnet"
            self.model = self._build_resnet(cfg.name, cfg.pretrained)
            if in_channels != 3:
                self._patch_first_conv(in_channels)
        elif cfg.name.lower().startswith("vit"):
            self.kind = "vit"
            self.model = self._build_vit(cfg.name, cfg.pretrained)
            if in_channels != 3:
                raise ValueError("torchvision ViT expects 3-channel input; set dataset channels=3 for vit backbones")
        else:
            raise ValueError(f"Unsupported backbone: {cfg.name}")

        self.out_layers = tuple(cfg.out_layers)

    def _build_resnet(self, name: str, pretrained: bool) -> nn.Module:
        name_l = name.lower()
        if name_l == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
            return torchvision.models.resnet18(weights=weights)
        if name_l == "resnet34":
            weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
            return torchvision.models.resnet34(weights=weights)
        if name_l == "resnet50":
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            return torchvision.models.resnet50(weights=weights)
        if name_l == "wide_resnet50_2":
            weights = torchvision.models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            return torchvision.models.wide_resnet50_2(weights=weights)
        raise ValueError(f"Unsupported ResNet variant: {name}")

    def _build_vit(self, name: str, pretrained: bool) -> nn.Module:
        name_l = name.lower()
        if name_l in {"vit_b_16", "vit"}:
            weights = torchvision.models.ViT_B_16_Weights.DEFAULT if pretrained else None
            return torchvision.models.vit_b_16(weights=weights)
        if name_l == "vit_l_16":
            weights = torchvision.models.ViT_L_16_Weights.DEFAULT if pretrained else None
            return torchvision.models.vit_l_16(weights=weights)
        raise ValueError(f"Unsupported ViT variant: {name}")

    def _patch_first_conv(self, in_channels: int) -> None:
        conv1: nn.Conv2d = self.model.conv1  # type: ignore[attr-defined]
        if conv1.in_channels == in_channels:
            return
        new_conv = nn.Conv2d(
            in_channels,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        )
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
            else:
                repeat = int((in_channels + 2) // 3)
                w = conv1.weight.repeat(1, repeat, 1, 1)[:, :in_channels]
                w = w * (3.0 / float(in_channels))
                new_conv.weight.copy_(w)
        self.model.conv1 = new_conv  # type: ignore[attr-defined]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.kind == "resnet":
            m = self.model
            x = m.conv1(x)
            x = m.bn1(x)
            x = m.relu(x)
            x = m.maxpool(x)

            outs: List[torch.Tensor] = []
            x = m.layer1(x)
            if "layer1" in self.out_layers:
                outs.append(x)
            x = m.layer2(x)
            if "layer2" in self.out_layers:
                outs.append(x)
            x = m.layer3(x)
            if "layer3" in self.out_layers:
                outs.append(x)
            x = m.layer4(x)
            if "layer4" in self.out_layers:
                outs.append(x)
            return outs

        m = self.model
        n, c, h, w = x.shape
        _ = c
        x = m._process_input(x)  # type: ignore[attr-defined]
        n_tokens = x.shape[1]
        cls_token = m.class_token.expand(n, -1, -1)  # type: ignore[attr-defined]
        x = torch.cat([cls_token, x], dim=1)
        x = m.encoder(x)  # type: ignore[attr-defined]
        x = x[:, 1:, :]
        patch = int(m.conv_proj.kernel_size[0])  # type: ignore[attr-defined]
        grid_h = int(h // patch)
        grid_w = int(w // patch)
        if grid_h * grid_w != n_tokens:
            grid_h = int((n_tokens) ** 0.5)
            grid_w = max(1, int(n_tokens // max(1, grid_h)))
        feat = x.transpose(1, 2).reshape(n, x.shape[-1], grid_h, grid_w)
        return [feat]


def upsample_to(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size_hw:
        return x
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
