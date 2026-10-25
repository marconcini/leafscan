"""Fine-tune on a fraction of PlantDoc's training images, evaluate on PlantDoc's test split.

    python -m leafscan.field_finetune --init plantvillage --fraction 0.25 --seed 0
    python -m leafscan.field_finetune --summary          # -> reports/field_finetune/summary.*

--init plantvillage starts from our PlantVillage checkpoint, --init imagenet from the ImageNet
weights that checkpoint was itself trained from. Every run uses the same number of optimiser
steps whatever the fraction, and the last checkpoint is evaluated (no selection on test).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from leafscan.data import build_transforms
from leafscan.field import DEFAULT_PLANTDOC_DIR, field_metrics, load_plantdoc
from leafscan.model import create_model, get_device, load_checkpoint
from leafscan.train import warmup_cosine

REPORT_DIR = Path("reports/field_finetune")


def stratified_fraction(items: list[dict], fraction: float, seed: int) -> list[dict]:
    """Keep `fraction` of each class (at least one image per class)."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for it in items:
        by_class[it["label"]].append(it)
    out = []
    for label in sorted(by_class):
        group = by_class[label][:]
        rng.shuffle(group)
        out += group[: max(1, round(len(group) * fraction))]
    return out


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the accuracy."""
    rng = np.random.default_rng(seed)
    accs = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


class ItemsDataset(Dataset):
    def __init__(self, items, class_to_idx, tf):
        self.items, self.class_to_idx, self.tf = items, class_to_idx, tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        it = self.items[i]
        return self.tf(Image.open(it["path"]).convert("RGB")), self.class_to_idx[it["label"]]


def init_model(init: str, pv_checkpoint: Path, device: torch.device):
    """Return (model, classes, img_size, model_name)."""
    _, ckpt = load_checkpoint(pv_checkpoint, torch.device("cpu"))
    classes, img_size, name = ckpt["classes"], ckpt["img_size"], ckpt["model_name"]
    model = create_model(name, len(classes), pretrained=(init == "imagenet"))
    if init == "plantvillage":
        model.load_state_dict(ckpt["state_dict"])
    elif init != "imagenet":
        raise ValueError(f"unknown init {init!r}")
    return model.to(device), classes, img_size, name


def run(init: str, fraction: float, seed: int, pv_checkpoint: Path,
        plantdoc_dir: Path = DEFAULT_PLANTDOC_DIR, steps: int = 400, batch_size: int = 32,
        lr: float = 1e-4, device: torch.device | None = None, num_workers: int = 4) -> dict:
    device = device or get_device()
    torch.manual_seed(seed)
    items, _ = load_plantdoc(plantdoc_dir)
    train_items = stratified_fraction([i for i in items if i["split"] == "train"], fraction, seed)
    test_items = [i for i in items if i["split"] == "test"]

    model, classes, img_size, name = init_model(init, pv_checkpoint, device)
    c2i = {c: i for i, c in enumerate(classes)}
    loader = DataLoader(ItemsDataset(train_items, c2i, build_transforms(img_size, train=True)),
                        batch_size=min(batch_size, len(train_items)), shuffle=True,
                        num_workers=num_workers, drop_last=True,
                        persistent_workers=num_workers > 0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(steps, steps // 10))

    model.train()
    step, losses = 0, []
    while step < steps:
        for x, y in loader:
            loss = criterion(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step >= steps:
                break

    model.eval()
    test_loader = DataLoader(ItemsDataset(test_items, c2i, build_transforms(img_size, False)),
                             batch_size=64, num_workers=num_workers)
    ys, probs = [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu())
            ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(probs).numpy()
    metrics = field_metrics(y, p, classes)
    lo, hi = bootstrap_ci(p.argmax(1) == y, seed=seed)
    return {"model": name, "init": init, "fraction": fraction, "seed": seed,
            "n_train": len(train_items), "steps": steps, "lr": lr,
            "final_train_loss": float(np.mean(losses[-20:])),
            "accuracy_ci95": [lo, hi], **metrics}


def summarize(report_dir: Path = REPORT_DIR) -> list[dict]:
    """Mean and std over seeds for each (init, fraction)."""
    groups = defaultdict(list)
    for f in sorted(report_dir.glob("*_seed*.json")):
        r = json.loads(f.read_text())
        groups[(r["init"], r["fraction"])].append(r)
    rows = []
    for (init, frac), rs in sorted(groups.items()):
        acc = np.array([r["accuracy"] for r in rs])
        rows.append({"init": init, "fraction": frac, "n_train": rs[0]["n_train"],
                     "seeds": len(rs), "accuracy_mean": float(acc.mean()),
                     "accuracy_std": float(acc.std(ddof=1)) if len(rs) > 1 else 0.0,
                     "balanced_mean": float(np.mean([r["balanced_accuracy"] for r in rs])),
                     "crop_mean": float(np.mean([r["crop_accuracy"] for r in rs]))})
    return rows


def plot_summary(rows: list[dict], zero_shot: float | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f2328", "#59636e", "#e6e8eb"
    colors = {"plantvillage": "#2f7d4f", "imagenet": "#5b6abf"}
    labels = {"plantvillage": "from PlantVillage model", "imagenet": "from ImageNet"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for init in ("plantvillage", "imagenet"):
        rs = [r for r in rows if r["init"] == init]
        if not rs:
            continue
        x = [r["n_train"] for r in rs]
        m = np.array([100 * r["accuracy_mean"] for r in rs])
        sd = np.array([100 * r["accuracy_std"] for r in rs])
        ax.plot(x, m, color=colors[init], linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=2, label=labels[init], zorder=3)
        ax.fill_between(x, m - sd, m + sd, color=colors[init], alpha=0.15, linewidth=0)
        ax.annotate(labels[init], (x[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=ink)
    ax.set_xscale("log")
    ns = sorted({r["n_train"] for r in rows})
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlim(ns[0] / 1.3, ns[-1] * 2.6)  # room for the direct labels
    if zero_shot is not None:
        ax.axhline(100 * zero_shot, color=muted, linewidth=1, linestyle="--")
        ax.text(ns[0], 100 * zero_shot + 1, f"zero-shot, no PlantDoc image: {100 * zero_shot:.1f}%",
                fontsize=8, color=muted, va="bottom")
        ax.set_ylim(bottom=100 * zero_shot - 4)
    ax.set_xlabel("PlantDoc training images (log scale)", color=muted)
    ax.set_ylabel("PlantDoc test accuracy (%)", color=muted)
    ax.set_title("ConvNeXt-T fine-tuned on field photos (mean ± std, 3 seeds)", color=ink,
                 loc="left", fontsize=11)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", choices=["plantvillage", "imagenet"], default="plantvillage")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--pv-checkpoint", type=Path, default=Path("models/convnext_tiny/best.pt"))
    ap.add_argument("--plantdoc-dir", type=Path, default=DEFAULT_PLANTDOC_DIR)
    ap.add_argument("--device", help="e.g. cpu (default: cuda > mps > cpu)")
    ap.add_argument("--summary", action="store_true", help="aggregate existing runs")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summary:
        rows = summarize()
        (REPORT_DIR / "summary.json").write_text(json.dumps(rows, indent=2))
        zs_path = Path("reports") / args.pv_checkpoint.parent.name / "field_plantdoc.json"
        zs = (json.loads(zs_path.read_text())["plantdoc_test_split_only"]["accuracy"]
              if zs_path.exists() else None)
        plot_summary(rows, zs, REPORT_DIR / "summary.png")
        for r in rows:
            print(f"{r['init']:12s} {r['fraction']:4.2f} (n={r['n_train']:4d}) "
                  f"acc {100 * r['accuracy_mean']:.1f} ± {100 * r['accuracy_std']:.1f}")
        return

    res = run(args.init, args.fraction, args.seed, args.pv_checkpoint, args.plantdoc_dir,
              steps=args.steps, device=torch.device(args.device) if args.device else None)
    out = REPORT_DIR / f"{args.init}_frac{args.fraction:.2f}_seed{args.seed}.json"
    out.write_text(json.dumps(res, indent=2))
    lo, hi = res["accuracy_ci95"]
    print(f"{out.stem}: n_train={res['n_train']} acc {res['accuracy']:.3f} "
          f"[{lo:.3f}, {hi:.3f}] crop {res['crop_accuracy']:.3f}")


if __name__ == "__main__":
    main()
