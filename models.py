# models.py
"""
Organized model factory for EE782 EuroSAT project.

Provides:
 - create_model_resnet(...)
 - create_model_densenet(...)
 - create_model_efficientnet(...)
 - create_model(name, ...)  # unified dispatcher (recommended)
 - count_parameters(model, verbose=False)

All factories use timm under the hood. They accept the same signature:
    (num_classes=10, pretrained=True, in_chans=3, freeze_backbone=False)

Examples:
    from models import create_model
    model = create_model("resnet50", num_classes=10, pretrained=True)

    from models import create_model_resnet
    model = create_model_resnet(num_classes=10, pretrained=True)
"""

from typing import Tuple
import timm
import torch.nn as nn


def _unfreeze_classifier_head(model):
    """
    Try to unfreeze the classifier head of a timm model by checking common attribute names.
    """
    # Common attribute names across timm models
    head_names = ("fc", "classifier", "head")
    unfroze = False

    for hn in head_names:
        if hasattr(model, hn):
            head = getattr(model, hn)
            try:
                for p in head.parameters():
                    p.requires_grad = True
                unfroze = True
            except Exception:
                # head might be a simple Tensor or other; skip safely
                pass

    # timm convenience: some models have get_classifier()
    try:
        cls = model.get_classifier()
        if cls is not None:
            for p in cls.parameters():
                p.requires_grad = True
            unfroze = True
    except Exception:
        pass

    return unfroze


def create_model_resnet(num_classes: int = 10,
                        pretrained: bool = True,
                        in_chans: int = 3,
                        freeze_backbone: bool = False,
                        variant: str = "resnet50") -> nn.Module:
    """
    Creates a ResNet model using timm.

    Args:
        num_classes: number of output classes.
        pretrained: load ImageNet pretrained weights if True.
        in_chans: number of input channels (3 for RGB, 13 for multispectral).
        freeze_backbone: if True, freeze backbone and leave classifier head trainable.
        variant: timm resnet variant (e.g., "resnet18","resnet34","resnet50","resnet101").

    Returns:
        PyTorch ResNet model ready for training.
    """
    model = timm.create_model(variant, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        _unfreeze_classifier_head(model)

    return model


def create_model_densenet(num_classes: int = 10,
                          pretrained: bool = True,
                          in_chans: int = 3,
                          freeze_backbone: bool = False,
                          variant: str = "densenet121") -> nn.Module:
    """
    Creates a DenseNet model using timm.

    Args:
        num_classes: number of output classes.
        pretrained: load ImageNet pretrained weights if True.
        in_chans: number of input channels (3 for RGB, 13 for multispectral).
        freeze_backbone: if True, freeze backbone and leave classifier head trainable.
        variant: timm densenet variant (e.g., "densenet121","densenet169","densenet201").

    Returns:
        PyTorch DenseNet model ready for training.
    """
    model = timm.create_model(variant, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        _unfreeze_classifier_head(model)

    return model


def create_model_efficientnet(num_classes: int = 10,
                              pretrained: bool = True,
                              in_chans: int = 3,
                              freeze_backbone: bool = False,
                              variant: str = "efficientnet_b0") -> nn.Module:
    """
    Creates an EfficientNet (or EfficientNet-family) model using timm.

    Args:
        num_classes: number of output classes.
        pretrained: load ImageNet pretrained weights if True.
        in_chans: number of input channels (3 for RGB, 13 for multispectral).
        freeze_backbone: if True, freeze backbone and leave classifier head trainable.
        variant: timm variant (e.g., "efficientnet_b0","efficientnet_b1", "efficientnet_b4", or efficientnetv2 variants).

    Returns:
        PyTorch EfficientNet model ready for training.
    """
    model = timm.create_model(variant, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        _unfreeze_classifier_head(model)

    return model


def create_model(name: str,
                 num_classes: int = 10,
                 pretrained: bool = True,
                 in_chans: int = 3,
                 freeze_backbone: bool = False) -> nn.Module:
    """
    Unified dispatcher that creates a model by name.

    Args:
        name: timm model string (e.g., "resnet50", "densenet121", "efficientnet_b0").
        num_classes, pretrained, in_chans, freeze_backbone: forwarded to the specific factory.

    Returns:
        PyTorch model ready for training.

    Notes:
        - Use this function in your train.py: create_model(name="resnet50", ...)
        - If you pass an unknown name, timm.create_model will raise an informative error.
    """
    # normalize name
    n = name.lower()
    # Simple dispatch based on keywords; fallback to timm.create_model directly if unknown
    try:
        if "resnet" in n:
            return create_model_resnet(num_classes=num_classes, pretrained=pretrained, in_chans=in_chans, freeze_backbone=freeze_backbone, variant=n)
        elif "densenet" in n:
            return create_model_densenet(num_classes=num_classes, pretrained=pretrained, in_chans=in_chans, freeze_backbone=freeze_backbone, variant=n)
        elif "efficientnet" in n or n.startswith("tf_efficientnet") or "efficientnetv2" in n:
            return create_model_efficientnet(num_classes=num_classes, pretrained=pretrained, in_chans=in_chans, freeze_backbone=freeze_backbone, variant=n)
        else:
            # Generic fallback — timm supports many models; just call it
            model = timm.create_model(n, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
            if freeze_backbone:
                for p in model.parameters():
                    p.requires_grad = False
                _unfreeze_classifier_head(model)
            return model
    except Exception as e:
        # Re-raise with extra context
        raise RuntimeError(f"Failed to create model '{name}': {e}") from e


def count_parameters(model: nn.Module, verbose: bool = False) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,} ({100.0 * trainable / total:.2f}%)")
    return total, trainable


# Quick sanity test when running directly
if __name__ == "__main__":
    # Basic smoke tests (no weights download if pretrained=False)
    m = create_model("resnet50", num_classes=10, pretrained=False)
    count_parameters(m, verbose=True)
    m = create_model("densenet121", num_classes=10, pretrained=False)
    count_parameters(m, verbose=True)
    m = create_model("efficientnet_b0", num_classes=10, pretrained=False)
    count_parameters(m, verbose=True)
