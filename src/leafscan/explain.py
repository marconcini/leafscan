"""Where do the models look? Grad-CAM analysis against leaf masks.

    python -m leafscan.explain metrics models/vit_small/best.pt [--random-weights]
    python -m leafscan.explain figure models/*/best.pt --images a.JPG b.JPG ...

`metrics` runs Grad-CAM on the whole test split and measures, for each image whose leaf
covers less than 90% of the frame (otherwise "leaf vs background" is meaningless):
- leaf focus: share of the CAM mass on the leaf, compared with the leaf area;
- pointing game: is the CAM maximum on the leaf?
`--random-weights` repeats it with a randomly initialised model (sanity check,
Adebayo et al. 2018).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from leafscan.compare import DISPLAY_NAMES  # noqa: E402
from leafscan.data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    Sample,
    build_transforms,
    leaf_mask,
    mask_transform,
    pretty_label,
    read_split,
)
from leafscan.gradcam import for_model, leaf_focus, pointing_hit  # noqa: E402
from leafscan.model import create_model, get_device, load_checkpoint  # noqa: E402

MAX_LEAF_AREA = 0.9


class MaskedDataset(Dataset):
    """Eval-transformed image, label, and leaf mask with the same geometry."""

    def __init__(self, samples: list[Sample], classes: list[str], img_size: int):
        self.samples = samples
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.tf = build_transforms(img_size, train=False)
        self.mask_tf = mask_transform(img_size)
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = self.tf(Image.open(s.path).convert("RGB"))
        m = leaf_mask(s.path)
        if m is None:
            mask, has_mask = torch.zeros(self.img_size, self.img_size), False
        else:
            mask = torch.from_numpy(np.array(self.mask_tf(Image.fromarray(m)))).float()
            has_mask = True
        return x, self.class_to_idx[s.label], mask, has_mask, idx


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    empty = [r for r in rows if np.isnan(r["focus"])]  # all-zero CAM: no positive evidence
    rows = [r for r in rows if not np.isnan(r["focus"])]
    if not rows:
        return {"n": 0, "n_empty_cam": len(empty)}
    focus = np.array([r["focus"] for r in rows])
    area = np.array([r["area"] for r in rows])
    return {
        "n": len(rows),
        "n_empty_cam": len(empty),
        "leaf_focus": float(focus.mean()),
        "leaf_area": float(area.mean()),
        "focus_minus_area": float((focus - area).mean()),
        "share_focus_above_area": float((focus > area).mean()),
        "pointing_game": float(np.mean([r["hit"] for r in rows])),
    }


def gradcam_metrics(ckpt_path: Path, samples: list[Sample], random_weights: bool = False,
                    batch_size: int = 32, num_workers: int = 4,
                    device: torch.device | None = None) -> dict:
    device = device or get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    if random_weights:
        model = create_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
        model.to(device).eval()
    cam_fn = for_model(model, ckpt["model_name"])
    loader = DataLoader(MaskedDataset(samples, ckpt["classes"], ckpt["img_size"]),
                        batch_size=batch_size, num_workers=num_workers)

    rows, n_no_mask, n_too_big = [], 0, 0
    for x, y, masks, has_mask, idx in loader:
        cams, logits = cam_fn(x.to(device))
        cams, pred = cams.cpu(), logits.argmax(1).cpu()
        for i in range(len(y)):
            if not has_mask[i]:
                n_no_mask += 1
                continue
            area = float(masks[i].mean())
            if area > MAX_LEAF_AREA:
                n_too_big += 1
                continue
            rows.append({"idx": int(idx[i]), "label": samples[int(idx[i])].label,
                         "correct": bool(pred[i] == y[i]), "area": area,
                         "focus": leaf_focus(cams[i], masks[i]),
                         "hit": pointing_hit(cams[i], masks[i])})
    cam_fn.remove()

    by_class = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r)
    return {
        "model": ckpt["model_name"],
        "random_weights": random_weights,
        "excluded_no_mask": n_no_mask,
        f"excluded_leaf_area_above_{MAX_LEAF_AREA}": n_too_big,
        "all": _summary(rows),
        "correct": _summary([r for r in rows if r["correct"]]),
        "wrong": _summary([r for r in rows if not r["correct"]]),
        "per_class": {c: _summary(rs) for c, rs in sorted(by_class.items())},
        "lowest_focus": [
            {"path": str(samples[r["idx"]].path), "focus": r["focus"], "area": r["area"]}
            for r in sorted(rows, key=lambda r: r["focus"] - r["area"])[:20]
        ],
    }


# --------------------------------------------------------------------------- figure


def _denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def figure(ckpts: list[Path], images: list[Path], out: Path,
           device: torch.device | None = None) -> None:
    """Rows: images. Columns: original + one Grad-CAM overlay per model (predicted class)."""
    device = device or get_device()
    fig, axes = plt.subplots(len(images), len(ckpts) + 1,
                             figsize=(2.2 * (len(ckpts) + 1), 2.3 * len(images)), squeeze=False)
    for col, ckpt_path in enumerate(ckpts, start=1):
        model, ckpt = load_checkpoint(ckpt_path, device)
        cam_fn = for_model(model, ckpt["model_name"])
        tf = build_transforms(ckpt["img_size"], train=False)
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in images])
        cams, logits = cam_fn(x.to(device))
        cam_fn.remove()
        probs = torch.softmax(logits, 1).cpu()
        for row, p in enumerate(images):
            ax = axes[row, col]
            ax.imshow(_denormalize(x[row]))
            ax.imshow(cams[row].cpu().numpy(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
            k = int(probs[row].argmax())
            ok = ckpt["classes"][k] == Path(p).parent.name
            ax.set_title(f"{DISPLAY_NAMES.get(ckpt_path.parent.name, ckpt_path.parent.name)}\n"
                         f"{'✓' if ok else '✗'} {pretty_label(ckpt['classes'][k])[:24]} "
                         f"{probs[row, k]:.2f}", fontsize=7, color="black" if ok else "firebrick")
            ax.axis("off")
            if col == 1:
                axes[row, 0].imshow(_denormalize(x[row]))
                axes[row, 0].set_title(f"label\n{pretty_label(Path(p).parent.name)[:30]}",
                                       fontsize=7)
                axes[row, 0].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("checkpoint", type=Path)
    m.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    m.add_argument("--reports-dir", type=Path, default=Path("reports"))
    m.add_argument("--random-weights", action="store_true")
    m.add_argument("--device", help="e.g. cpu (default: cuda > mps > cpu)")
    f = sub.add_parser("figure")
    f.add_argument("checkpoints", type=Path, nargs="+")
    f.add_argument("--images", type=Path, nargs="+", required=True)
    f.add_argument("--out", type=Path, default=Path("reports/gradcam_examples.jpg"))
    f.add_argument("--device", help="e.g. cpu (default: cuda > mps > cpu)")
    args = p.parse_args()

    if args.cmd == "figure":
        figure(args.checkpoints, args.images, args.out,
               device=torch.device(args.device) if args.device else None)
        return
    res = gradcam_metrics(args.checkpoint, read_split(args.data_dir, "test"), args.random_weights,
                          device=torch.device(args.device) if args.device else None)
    run = args.checkpoint.parent.name
    name = "gradcam_random.json" if args.random_weights else "gradcam.json"
    out = args.reports_dir / run / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    a = res["all"]
    print(f"{run}{' (random weights)' if args.random_weights else ''}: n={a['n']} "
          f"leaf focus {a['leaf_focus']:.3f} vs area {a['leaf_area']:.3f} "
          f"pointing {a['pointing_game']:.3f} | wrong: {res['wrong']}")


if __name__ == "__main__":
    main()
