"""Do the models use the background? Causal test by editing the background of test images.

    python -m leafscan.background models/convnext_tiny/best.pt [--device cpu]
    python -m leafscan.background --figure                     # illustrate the conditions

Conditions (leaf masks from PlantVillage's segmented variant):
  original         unchanged image
  leaf_only        background replaced by a neutral grey
  background_only  leaf (mask dilated by a few px) replaced by grey
  background_only_box  same, but the whole bounding box of the leaf is greyed out, so the
                   grey hole no longer reveals the leaf's shape
  swap             leaf pasted on the background of an image from another class
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from leafscan.data import (
    DEFAULT_DATA_DIR,
    IMAGENET_MEAN,
    Sample,
    build_transforms,
    leaf_mask,
    read_split,
)
from leafscan.model import get_device, load_checkpoint

CONDITIONS = ("original", "leaf_only", "background_only", "background_only_box", "swap")
FILL = np.array([round(255 * c) for c in IMAGENET_MEAN], dtype=np.uint8)  # zero after normalising
MAX_LEAF_AREA = 0.9
DILATE_PX = 8


def leaf_only(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[~mask] = FILL
    return out


def background_only(img: np.ndarray, mask: np.ndarray, dilate: int = DILATE_PX) -> np.ndarray:
    out = img.copy()
    square = np.ones((3, 3), bool)  # dilate evenly, diagonals included
    hole = ndimage.binary_dilation(mask, square, iterations=dilate) if dilate else mask
    out[hole] = FILL
    return out


def background_only_box(img: np.ndarray, mask: np.ndarray, dilate: int = DILATE_PX
                        ) -> np.ndarray:
    """Background only, with the leaf's bounding box greyed out (hides the leaf silhouette)."""
    out = img.copy()
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(rows):
        h, w = mask.shape
        out[max(0, rows[0] - dilate):min(h, rows[-1] + dilate + 1),
            max(0, cols[0] - dilate):min(w, cols[-1] + dilate + 1)] = FILL
    return out


def swap(img: np.ndarray, mask: np.ndarray, donor: np.ndarray, donor_mask: np.ndarray,
         dilate: int = DILATE_PX) -> np.ndarray:
    """Leaf of `img` on the background of `donor` (whose own leaf is greyed out)."""
    out = background_only(donor, donor_mask, dilate)
    out[mask] = img[mask]
    return out


def eligible(samples: list[Sample]) -> list[tuple[Sample, np.ndarray]]:
    """Samples with a leaf mask whose leaf covers at most MAX_LEAF_AREA of the frame."""
    out = []
    for s in samples:
        m = leaf_mask(s.path)
        if m is not None and m.mean() <= MAX_LEAF_AREA:
            out.append((s, m))
    return out


def pick_donors(items: list[tuple[Sample, np.ndarray]], seed: int = 0) -> list[int]:
    """For each item, the index of a random item from a different class."""
    rng = random.Random(seed)
    donors = []
    for s, _ in items:
        while True:
            j = rng.randrange(len(items))
            if items[j][0].label != s.label:
                donors.append(j)
                break
    return donors


class EditedDataset(Dataset):
    def __init__(self, items, classes, img_size, condition, donors=None):
        self.items, self.condition, self.donors = items, condition, donors
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.tf = build_transforms(img_size, train=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        s, mask = self.items[i]
        img = np.array(Image.open(s.path).convert("RGB"))
        donor_label = -1
        if self.condition == "leaf_only":
            img = leaf_only(img, mask)
        elif self.condition == "background_only":
            img = background_only(img, mask)
        elif self.condition == "background_only_box":
            img = background_only_box(img, mask)
        elif self.condition == "swap":
            d, d_mask = self.items[self.donors[i]]
            img = swap(img, mask, np.array(Image.open(d.path).convert("RGB")), d_mask)
            donor_label = self.class_to_idx[d.label]
        return self.tf(Image.fromarray(img)), self.class_to_idx[s.label], donor_label


@torch.no_grad()
def run_condition(model, items, classes, img_size, condition, device, donors=None,
                  batch_size=64, num_workers=4) -> dict:
    loader = DataLoader(EditedDataset(items, classes, img_size, condition, donors),
                        batch_size=batch_size, num_workers=num_workers)
    y, pred, donor = [], [], []
    for x, yy, dd in loader:
        pred.append(model(x.to(device)).argmax(1).cpu())
        y.append(yy)
        donor.append(dd)
    y, pred, donor = torch.cat(y).numpy(), torch.cat(pred).numpy(), torch.cat(donor).numpy()

    per_class = defaultdict(list)
    for t, p in zip(y, pred, strict=True):
        per_class[classes[t]].append(p == t)
    res = {"n": len(y), "accuracy": float((pred == y).mean()),
           "per_class_accuracy": {c: float(np.mean(v)) for c, v in sorted(per_class.items())}}
    if condition == "swap":
        res["follows_leaf"] = res.pop("accuracy")
        res["follows_background"] = float((pred == donor).mean())
        res["neither"] = 1 - res["follows_leaf"] - res["follows_background"]
    return res


def evaluate(ckpt_path: Path, samples: list[Sample], device: torch.device | None = None,
             num_workers: int = 4) -> dict:
    device = device or get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    items = eligible(samples)
    donors = pick_donors(items)
    out = {"model": ckpt["model_name"], "chance": 1 / len(ckpt["classes"]),
           "fill_rgb": FILL.tolist(), "dilate_px": DILATE_PX}
    for cond in CONDITIONS:
        out[cond] = run_condition(model, items, ckpt["classes"], ckpt["img_size"], cond, device,
                                  donors=donors, num_workers=num_workers)
    return out


def figure(samples: list[Sample], out: Path, n: int = 3, seed: int = 3) -> None:
    """A few test images under each condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    items = eligible(rng.sample(samples, 60))
    picks = rng.sample(range(len(items)), n)
    donors = pick_donors(items, seed)
    fig, axes = plt.subplots(n, 5, figsize=(11, 2.4 * n), squeeze=False)
    titles = ["A. original", "B. leaf only", "C. background only", "C'. background only (box)",
              "D. swapped background"]
    for r, i in enumerate(picks):
        s, m = items[i]
        img = np.array(Image.open(s.path).convert("RGB"))
        d, dm = items[donors[i]]
        panels = [img, leaf_only(img, m), background_only(img, m), background_only_box(img, m),
                  swap(img, m, np.array(Image.open(d.path).convert("RGB")), dm)]
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--device", help="e.g. cpu (default: cuda > mps > cpu)")
    p.add_argument("--figure", action="store_true", help="draw reports/background_conditions.jpg")
    args = p.parse_args()

    samples = read_split(args.data_dir, "test")
    if args.figure:
        figure(samples, args.reports_dir / "background_conditions.jpg")
        return
    if args.checkpoint is None:
        p.error("checkpoint is required unless --figure is given")
    res = evaluate(args.checkpoint, samples,
                   torch.device(args.device) if args.device else None)
    out = args.reports_dir / args.checkpoint.parent.name / "background.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    sw = res["swap"]
    print(f"{args.checkpoint.parent.name}: original {res['original']['accuracy']:.4f} | "
          f"leaf only {res['leaf_only']['accuracy']:.4f} | "
          f"background only {res['background_only']['accuracy']:.4f} "
          f"(box {res['background_only_box']['accuracy']:.4f}) | "
          f"swap: leaf {sw['follows_leaf']:.4f} background {sw['follows_background']:.4f} "
          f"(n={res['original']['n']})")


if __name__ == "__main__":
    main()
