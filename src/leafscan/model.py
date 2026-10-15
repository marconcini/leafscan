"""Model creation, device selection and checkpoint I/O."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Any timm backbone with a fresh classification head (transfer learning)."""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def save_checkpoint(path: Path, model: nn.Module, model_name: str, classes: list[str],
                    img_size: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "classes": classes,
            "img_size": img_size,
            **(extra or {}),
        },
        path,
    )


def load_checkpoint(path: Path | str, device: torch.device | None = None
                    ) -> tuple[nn.Module, dict]:
    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
