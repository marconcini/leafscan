"""Lab -> field: evaluate PlantVillage models on PlantDoc field photos [Singh et al. 2020].

    git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git data/plantdoc
    python -m leafscan.field models/vit_small/best.pt [--device cpu]

No model was trained on PlantDoc, so every PlantDoc image (train and test folders) is
unseen: all are used for this zero-shot evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from leafscan.data import build_transforms
from leafscan.model import get_device, load_checkpoint

DEFAULT_PLANTDOC_DIR = Path("data/plantdoc")

# PlantDoc folder name -> PlantVillage class
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}


def crop_of(label: str) -> str:
    return label.split("___", 1)[0]


def is_healthy(label: str) -> bool:
    return label.split("___", 1)[-1] == "healthy"


def load_plantdoc(root: Path = DEFAULT_PLANTDOC_DIR) -> tuple[list[dict], list[str]]:
    """All readable PlantDoc images as {path, label (PlantVillage), split}, plus skipped files.

    PlantDoc contains byte-identical duplicates ('CAR1.jpg' / 'car1.jpg'), some of them filed
    under *different* classes. Duplicates with the same label are kept once; duplicates with
    conflicting labels are all dropped, since their true class is unknown.
    """
    by_hash: dict[str, list[dict]] = {}
    skipped = []
    for split in ("train", "test"):
        for folder in sorted((root / split).iterdir()):
            if not folder.is_dir():
                continue
            if folder.name not in PLANTDOC_TO_PLANTVILLAGE:
                raise KeyError(f"Unmapped PlantDoc class: {folder.name!r}")
            for f in sorted(folder.iterdir()):
                if f.name.startswith("."):
                    continue
                try:
                    with Image.open(f) as im:
                        im.verify()
                except Exception:  # corrupted or not an image
                    skipped.append(f"{f} (unreadable)")
                    continue
                digest = hashlib.md5(f.read_bytes()).hexdigest()
                by_hash.setdefault(digest, []).append(
                    {"path": f, "label": PLANTDOC_TO_PLANTVILLAGE[folder.name], "split": split})

    items = []
    for copies in by_hash.values():
        labels = {c["label"] for c in copies}
        if len(labels) > 1:
            skipped += [f"{c['path']} (conflicting labels: {sorted(labels)})" for c in copies]
            continue
        # prefer the copy in the official test split, to keep that split as complete as possible
        copies = sorted(copies, key=lambda c: c["split"] != "test")
        items.append(copies[0])
        skipped += [f"{c['path']} (duplicate of {copies[0]['path']})" for c in copies[1:]]
    return items, skipped


class FieldDataset(Dataset):
    def __init__(self, items, class_to_idx, img_size):
        self.items, self.class_to_idx = items, class_to_idx
        self.tf = build_transforms(img_size, train=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        return self.tf(Image.open(it["path"]).convert("RGB")), self.class_to_idx[it["label"]]


def field_metrics(y: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == y
    true_l, pred_l = [classes[i] for i in y], [classes[i] for i in pred]
    per_class = defaultdict(list)
    for t, ok in zip(true_l, correct, strict=True):
        per_class[t].append(bool(ok))
    confusions = Counter((t, p) for t, p, ok in zip(true_l, pred_l, correct, strict=True) if not ok)
    return {
        "n": len(y),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean([np.mean(v) for v in per_class.values()])),
        "crop_accuracy": float(np.mean([crop_of(t) == crop_of(p)
                                        for t, p in zip(true_l, pred_l, strict=True)])),
        "health_accuracy": float(np.mean([is_healthy(t) == is_healthy(p)
                                          for t, p in zip(true_l, pred_l, strict=True)])),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_wrong": float(conf[~correct].mean()) if (~correct).any() else None,
        "share_wrong_with_confidence_above_0.9": float((conf[~correct] > 0.9).mean())
        if (~correct).any() else None,
        "most_predicted": Counter(pred_l).most_common(5),
        "per_class": {c: {"n": len(v), "accuracy": float(np.mean(v))}
                      for c, v in sorted(per_class.items())},
        "top_confusions": [{"true": t, "pred": p, "count": n}
                           for (t, p), n in confusions.most_common(10)],
    }


@torch.no_grad()
def evaluate(ckpt_path: Path, items: list[dict], device: torch.device | None = None,
             batch_size: int = 64, num_workers: int = 4) -> dict:
    device = device or get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    classes = ckpt["classes"]
    loader = DataLoader(FieldDataset(items, {c: i for i, c in enumerate(classes)},
                                     ckpt["img_size"]),
                        batch_size=batch_size, num_workers=num_workers)
    ys, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(probs).numpy()
    res = {"model": ckpt["model_name"], "all": field_metrics(y, p, classes)}
    test_idx = np.array([it["split"] == "test" for it in items])
    res["plantdoc_test_split_only"] = field_metrics(y[test_idx], p[test_idx], classes)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--plantdoc-dir", type=Path, default=DEFAULT_PLANTDOC_DIR)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--device", help="e.g. cpu (default: cuda > mps > cpu)")
    args = ap.parse_args()

    items, skipped = load_plantdoc(args.plantdoc_dir)
    res = evaluate(args.checkpoint, items, torch.device(args.device) if args.device else None)
    res["skipped_unreadable"] = skipped
    out = args.reports_dir / args.checkpoint.parent.name / "field_plantdoc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    a = res["all"]
    print(f"{args.checkpoint.parent.name}: n={a['n']} acc {a['accuracy']:.3f} "
          f"bal {a['balanced_accuracy']:.3f} crop {a['crop_accuracy']:.3f} "
          f"health {a['health_accuracy']:.3f} conf wrong {a['mean_confidence_wrong']:.2f} "
          f"(skipped {len(skipped)})")


if __name__ == "__main__":
    main()
