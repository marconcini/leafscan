"""Export a checkpoint for sharing: weights as safetensors + a JSON config.

    python -m leafscan.export models/leafscan_field/best.pt hub/leafscan-convnext-tiny

A .pt checkpoint is a pickle, which can run code when loaded; safetensors only holds tensors,
so it is the safe format to publish. `load_exported` accepts a local folder or a Hugging Face
Hub repo id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from leafscan.model import create_model, load_checkpoint

CONFIG_KEYS = ("model_name", "classes", "img_size", "temperature", "threshold",
               "threshold_target_accuracy", "trained_on")


def export(ckpt_path: Path, out_dir: Path) -> dict:
    model, ckpt = load_checkpoint(ckpt_path, torch.device("cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, out_dir / "model.safetensors")
    config = {k: ckpt[k] for k in CONFIG_KEYS if k in ckpt}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return config


def load_exported(source: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load from a local export folder, or download it from the Hugging Face Hub."""
    folder = Path(source)
    if not folder.is_dir():
        from huggingface_hub import snapshot_download

        folder = Path(snapshot_download(str(source),
                                        allow_patterns=["model.safetensors", "config.json"]))
    config = json.loads((folder / "config.json").read_text())
    model = create_model(config["model_name"], len(config["classes"]), pretrained=False)
    model.load_state_dict(load_file(folder / "model.safetensors"))
    return model.to(device).eval(), config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    config = export(args.checkpoint, args.out_dir)
    print(f"exported {config['model_name']} ({len(config['classes'])} classes) to {args.out_dir}")


if __name__ == "__main__":
    main()
