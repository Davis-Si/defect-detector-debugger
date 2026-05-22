"""Model: ResNet18 backbone (pretrained, frozen) + linear head.

We freeze the backbone because:
  1. CPU training of a full ResNet on 1440 images is wasteful.
  2. ImageNet features generalise well enough on this dataset that the
     bottleneck is the *data*, not the model — which is exactly the failure
     mode this project is designed to expose and analyse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

from .data import NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(*, train: bool, augment: str = "none") -> T.Compose:
    """augment ∈ {'none', 'flip', 'flip_rotate', 'flip_rotate_mild'}."""
    base = [T.Resize((224, 224))]
    if train and augment != "none":
        base.append(T.RandomHorizontalFlip(p=0.5))
        if augment == "flip_rotate":
            base.append(T.RandomRotation(degrees=15))
        elif augment == "flip_rotate_mild":
            base.append(T.RandomRotation(degrees=5))
    base.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(base)


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    backbone = tvm.resnet18(weights=weights)
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return penultimate-layer features (512-d for ResNet18)."""
    # We re-run the forward up to global avgpool. Cheaper than registering hooks.
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)
