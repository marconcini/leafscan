"""Inference for the demo: calibrated top-k, abstention, Grad-CAM overlay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from leafscan.data import IMAGENET_MEAN, IMAGENET_STD, build_transforms, pretty_label
from leafscan.export import load_exported
from leafscan.gradcam import for_model
from leafscan.model import load_checkpoint


@dataclass
class Prediction:
    top: list[tuple[str, float]]  # (class name, calibrated probability), best first
    confident: bool  # top probability >= the field-calibrated threshold
    threshold: float
    heatmap: np.ndarray | None = field(default=None, repr=False)  # (H, W) in [0, 1]

    @property
    def label(self) -> str:
        return self.top[0][0]

    @property
    def verdict(self) -> str:
        name, p = self.top[0]
        if self.confident:
            return f"**{pretty_label(name)}** (confidence {p:.0%})"
        return (f"**Not sure.** Best guess: {pretty_label(name)} ({p:.0%}), below the "
                f"{self.threshold:.0%} confidence needed to answer. Try a closer photo of a "
                f"single leaf, or ask an expert.")


class Predictor:
    def __init__(self, source: Path | str, device: str = "cpu"):
        """`source`: a .pt checkpoint, an exported folder, or a Hugging Face Hub repo id."""
        self.device = torch.device(device)
        if str(source).endswith(".pt"):
            self.model, ckpt = load_checkpoint(source, self.device)
        else:
            self.model, ckpt = load_exported(source, device)
        self.classes = ckpt["classes"]
        self.img_size = ckpt["img_size"]
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.threshold = float(ckpt.get("threshold", 0.0))
        self.model_name = ckpt["model_name"]
        self.tf = build_transforms(self.img_size, train=False)

    def __call__(self, image: Image.Image, k: int = 3, explain: bool = True) -> Prediction:
        x = self.tf(image.convert("RGB"))[None].to(self.device)
        if explain:
            cam_fn = for_model(self.model, self.model_name)
            cams, logits = cam_fn(x)
            cam_fn.remove()
            heatmap = cams[0].cpu().numpy()
        else:
            with torch.no_grad():
                logits = self.model(x)
            heatmap = None
        probs = torch.softmax(logits[0].float().cpu() / self.temperature, 0).numpy()
        order = np.argsort(-probs)[:k]
        top = [(self.classes[i], float(probs[i])) for i in order]
        return Prediction(top, top[0][1] >= self.threshold, self.threshold, heatmap)

    def overlay(self, image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
        """The image as the model saw it (resized + centre crop), with the Grad-CAM map."""
        import matplotlib

        x = self.tf(image.convert("RGB"))
        mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        std = torch.tensor(IMAGENET_STD)[:, None, None]
        rgb = (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        colored = matplotlib.colormaps["jet"](heatmap)[..., :3]
        blended = (1 - alpha) * rgb + alpha * colored
        return Image.fromarray((blended * 255).astype(np.uint8))
