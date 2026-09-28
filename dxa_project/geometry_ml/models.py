"""Compact pretrained backbones and spatial heads for DXA geometry."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18, ResNet50_Weights, resnet50
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from types import MethodType


def _empty_safe_roi_forward(self, features, proposals, image_shapes, targets=None):
    # A negative image can legitimately have no RPN proposals. Cross entropy
    # over zero sampled RoIs is undefined; only the RPN has supervision here.
    if (self.training and targets is not None
            and all(len(p) == 0 for p in proposals)
            and all(len(t['boxes']) == 0 for t in targets)):
        zero = sum(f.sum() * 0 for f in features.values())
        self.empty_negative_batches = getattr(self, 'empty_negative_batches', 0) + 1
        return [], {'loss_classifier': zero, 'loss_box_reg': zero}
    return RoIHeads.forward(self, features, proposals, image_shapes, targets)


def _safe_detector(model):
    model.roi_heads.forward = MethodType(_empty_safe_roi_forward, model.roi_heads)
    return model


class Router(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.net.fc = nn.Sequential(nn.LayerNorm(self.net.fc.in_features),
                                    nn.Linear(self.net.fc.in_features, 3))

    def forward(self, x):
        return self.net(x)

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen ImageNet encoder must keep its BatchNorm statistics too.
        self.net.eval()
        self.net.fc.train(mode)
        return self


class Up(nn.Module):
    def __init__(self, incoming: int, skip: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming + skip, outgoing, 3, padding=1),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.layers(torch.cat((x, skip), dim=1) if skip is not None else x)


class SpatialNet(nn.Module):
    """ResNet18 encoder with a full-resolution U-Net decoder."""

    def __init__(self, region: str, pretrained: bool = True, backbone: str = 'resnet18'):
        super().__init__()
        if region not in ("SPINE", "HIP"):
            raise ValueError(region)
        self.region = region
        if backbone not in ('resnet18','resnet50'):
            raise ValueError(backbone)
        encoder = (resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None) if backbone=='resnet18'
                   else resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None))
        channels_stage=[64,128,256,512] if backbone=='resnet18' else [256,512,1024,2048]
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.up3 = Up(channels_stage[3], channels_stage[2], 256)
        self.up2 = Up(256, channels_stage[1], 128)
        self.up1 = Up(128, channels_stage[0], 64)
        self.up0 = Up(64, 64, 32)
        self.up_final = Up(32, 0, 16)
        channels = 3 if region == "SPINE" else 4
        self.spatial = nn.Conv2d(16, channels, 1)
        self.presence = nn.Linear(channels_stage[3], 2 if region == "SPINE" else 3)
        self.roi = nn.Linear(channels_stage[3], 3) if region == "HIP" else None

    def forward(self, x):
        a = self.stem(x)
        b = self.layer1(self.pool(a))
        c = self.layer2(b)
        d = self.layer3(c)
        e = self.layer4(d)
        pooled = F.adaptive_avg_pool2d(e, 1).flatten(1)
        decoded = self.up_final(self.up0(self.up1(self.up2(self.up3(e, d), c), b), a))
        result = {"spatial": self.spatial(decoded), "presence": self.presence(pooled)}
        if self.roi is not None:
            result["roi"] = torch.sigmoid(self.roi(pooled))
        return result


def artifact_detector(pretrained: bool = True, heavy: bool = False):
    if heavy:
        model=fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
                                        weights_backbone=None,trainable_backbone_layers=3)
        model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
        return _safe_detector(model)
    weights = (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
               if pretrained else None)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, weights_backbone=None,
        trainable_backbone_layers=2 if pretrained else 6)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2)
    return _safe_detector(model)


def dice_bce(logits: torch.Tensor, truth: torch.Tensor,
             positive_weight: float = 8.0) -> torch.Tensor:
    weight = torch.tensor(positive_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, truth, pos_weight=weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * truth).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1))
    return bce + dice.mean()


def landmark_loss(logits: torch.Tensor, truth: torch.Tensor,
                  visible: torch.Tensor) -> torch.Tensor:
    """Foreground-weighted heatmaps plus differentiable coordinate error."""
    probability = torch.sigmoid(logits)
    pixel = ((probability - truth).square() * (1 + 100 * truth)).mean()
    batch, channels, height, width = logits.shape
    distribution = torch.softmax(logits.flatten(2), dim=-1).view_as(logits)
    x_grid = torch.linspace(0, 1, width, device=logits.device)[None, None, None, :]
    y_grid = torch.linspace(0, 1, height, device=logits.device)[None, None, :, None]
    predicted_x = (distribution * x_grid).sum(dim=(-2, -1))
    predicted_y = (distribution * y_grid).sum(dim=(-2, -1))
    mass = truth.sum(dim=(-2, -1)).clamp_min(1e-6)
    target_x = (truth * x_grid).sum(dim=(-2, -1)) / mass
    target_y = (truth * y_grid).sum(dim=(-2, -1)) / mass
    coordinate = ((predicted_x - target_x).abs() +
                  (predicted_y - target_y).abs()) * visible
    return 10 * pixel + 5 * coordinate.sum() / visible.sum().clamp_min(1)


def spatial_loss(outputs: dict, targets: list[dict], region: str) -> torch.Tensor:
    device = outputs["spatial"].device
    if region == "SPINE":
        lines = torch.stack([t["line"] for t in targets]).to(device)
        points = torch.stack([t["crest"] for t in targets]).to(device)
        present = torch.stack([t["crest_present"] for t in targets]).to(device)
        return (dice_bce(outputs["spatial"][:, :1], lines, 3) +
                landmark_loss(outputs["spatial"][:, 1:], points, present) +
                F.binary_cross_entropy_with_logits(outputs["presence"], present))
    points = torch.stack([t["hip_points"] for t in targets]).to(device)
    present = torch.stack([t["hip_present"] for t in targets]).to(device)
    mask = torch.stack([t["trochanter"] for t in targets]).to(device)
    roi = torch.stack([t["roi"] for t in targets]).to(device)
    roi_valid = torch.stack([t["roi_present"] for t in targets]).to(device).bool()
    roi_loss = (F.smooth_l1_loss(outputs["roi"][roi_valid], roi[roi_valid])
                if roi_valid.any() else outputs["roi"].sum() * 0)
    order_loss = F.relu(outputs["roi"][:, 0] - outputs["roi"][:, 1] + .02).mean()
    return (landmark_loss(outputs["spatial"][:, :3], points, present) +
            F.binary_cross_entropy_with_logits(outputs["presence"], present) +
            dice_bce(outputs["spatial"][:, 3:], mask, 5) +
            5 * roi_loss + order_loss)


def detector_images(images: torch.Tensor) -> list[torch.Tensor]:
    """Torchvision detectors perform their own ImageNet normalization."""
    mean = torch.tensor((.485, .456, .406), device=images.device)[None, :, None, None]
    std = torch.tensor((.229, .224, .225), device=images.device)[None, :, None, None]
    return list((images * std + mean).clamp(0, 1))
