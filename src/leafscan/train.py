"""Fine-tune a pretrained backbone on PlantVillage.

    python -m leafscan.train --config configs/efficientnet_b0.yaml
    python -m leafscan.train --config configs/efficientnet_b0.yaml --limit-per-class 20 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import timm
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from leafscan.data import (
    DEFAULT_DATA_DIR,
    PlantVillageDataset,
    build_transforms,
    load_splits,
    subsample,
)
from leafscan.evaluate import evaluate_checkpoint
from leafscan.model import create_model, get_device, save_checkpoint

DEFAULTS = {
    "model": "efficientnet_b0.ra_in1k",
    "pretrained": True,
    "img_size": 224,
    "epochs": 5,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "warmup_epochs": 0.5,
    "label_smoothing": 0.1,
    "val_fraction": 0.1,
    "num_workers": 4,
    "seed": 42,
    "limit_per_class": None,
}


def run_info(device: torch.device) -> dict:
    """What is needed to reproduce a run: code version, libraries, hardware."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"git_commit": commit, "python": platform.python_version(),
            "torch": str(torch.__version__), "timm": str(timm.__version__), "device": str(device)}


def make_loader(samples, classes, img_size, train, cfg) -> DataLoader:
    ds = PlantVillageDataset(samples, classes, build_transforms(img_size, train=train))
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=train,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
        drop_last=train and len(ds) > cfg["batch_size"],
    )


def warmup_cosine(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return fn


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return {"val_loss": loss_sum / n, "val_acc": correct / n}


def train(cfg: dict, data_dir: Path, out_dir: Path) -> Path:
    torch.manual_seed(cfg["seed"])
    device = get_device()
    train_s, val_s, test_s, classes = load_splits(data_dir, cfg["val_fraction"], cfg["seed"])
    if cfg["limit_per_class"]:
        train_s = subsample(train_s, cfg["limit_per_class"], cfg["seed"])
        val_s = subsample(val_s, max(2, cfg["limit_per_class"] // 4), cfg["seed"])
        test_s = subsample(test_s, max(2, cfg["limit_per_class"] // 4), cfg["seed"])
    print(f"device={device} classes={len(classes)} "
          f"train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    train_loader = make_loader(train_s, classes, cfg["img_size"], True, cfg)
    val_loader = make_loader(val_s, classes, cfg["img_size"], False, cfg)

    model = create_model(cfg["model"], len(classes), cfg["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine(cfg["epochs"] * steps_per_epoch,
                      int(cfg["warmup_epochs"] * steps_per_epoch)),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "run_info": run_info(device)}))
    ckpt_path = out_dir / "best.pt"
    history, best_acc = [], -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {loss_sum / n:.4f}")

        metrics = {"epoch": epoch, "train_loss": loss_sum / n,
                   **validate(model, val_loader, criterion, device),
                   "time_s": round(time.time() - t0, 1)}
        history.append(metrics)
        print(json.dumps(metrics))
        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            save_checkpoint(ckpt_path, model, cfg["model"], classes, cfg["img_size"],
                            {"epoch": epoch, "val_acc": best_acc})

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val_acc={best_acc:.4f} -> {ckpt_path}")

    report = evaluate_checkpoint(ckpt_path, test_s, out_dir, batch_size=cfg["batch_size"],
                                 num_workers=cfg["num_workers"])
    print(f"Test accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f}")
    return ckpt_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--run-name")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--limit-per-class", type=int, help="subsample for a quick smoke run")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(args.config.read_text()) or {})
    for key in ("epochs", "batch_size", "limit_per_class"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    run_name = args.run_name or cfg["model"].split(".")[0]
    train(cfg, args.data_dir, Path("models") / run_name)


if __name__ == "__main__":
    main()
