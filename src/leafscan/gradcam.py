"""Grad-CAM for CNNs and Vision Transformers [Selvaraju et al. 2017].

Grad-CAM weights each channel of a feature map by the mean gradient of the class
score w.r.t. that channel, sums the weighted channels and keeps the positive part:
regions that increase the score of the class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Last spatial feature map of each supported architecture."""
    name = model_name.split(".")[0]
    if name.startswith("vit"):
        return model.blocks[-1].norm1  # patch tokens entering the last attention block
    if name.startswith("resnet"):
        return model.layer4
    if name.startswith("convnext"):
        return model.stages[-1]
    if name.startswith("efficientnet"):
        return model.bn2  # BN + activation after the last 1x1 conv
    raise ValueError(f"No Grad-CAM target layer defined for {model_name!r}")


def tokens_to_map(t: torch.Tensor, num_prefix_tokens: int) -> torch.Tensor:
    """(B, N, C) transformer tokens -> (B, C, h, w), dropping class/register tokens."""
    t = t[:, num_prefix_tokens:]
    side = int(round(t.shape[1] ** 0.5))
    if side * side != t.shape[1]:
        raise ValueError(f"{t.shape[1]} patch tokens do not form a square grid")
    return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], side, side)


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module, num_prefix_tokens: int = 0):
        self.model = model
        self.num_prefix_tokens = num_prefix_tokens
        self._acts: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._save)

    def _save(self, _module, _inputs, output):
        output.retain_grad()
        self._acts = output

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cams, logits). cams: (B, H, W) in [0, 1], same size as x.

        class_idx defaults to the predicted class.
        """
        self.model.zero_grad(set_to_none=True)
        # grad on the input guarantees the hooked activation is part of the graph,
        # even if the model's weights are frozen
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = logits.argmax(1)
            logits.gather(1, class_idx[:, None]).sum().backward()

        acts, grads = self._acts, self._acts.grad
        if acts.ndim == 3:  # transformer tokens
            acts = tokens_to_map(acts, self.num_prefix_tokens)
            grads = tokens_to_map(grads, self.num_prefix_tokens)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        flat = cam.flatten(1)
        lo, hi = flat.min(1).values[:, None, None], flat.max(1).values[:, None, None]
        cam = torch.where(hi > lo, (cam - lo) / (hi - lo).clamp(min=1e-12), torch.zeros_like(cam))
        return cam.detach(), logits.detach()


def for_model(model: nn.Module, model_name: str) -> GradCAM:
    return GradCAM(model, target_layer(model, model_name),
                   num_prefix_tokens=getattr(model, "num_prefix_tokens", 0))


# --------------------------------------------------------------------------- metrics


def leaf_focus(cam: torch.Tensor, mask: torch.Tensor) -> float:
    """Share of the Grad-CAM mass that lies on the leaf (uniform map -> leaf area)."""
    total = cam.sum()
    return float((cam * mask).sum() / total) if total > 0 else float("nan")


def pointing_hit(cam: torch.Tensor, mask: torch.Tensor) -> bool:
    """Pointing game [Zhang et al. 2016]: is the CAM maximum on the leaf?"""
    return bool(mask.flatten()[cam.flatten().argmax()])
