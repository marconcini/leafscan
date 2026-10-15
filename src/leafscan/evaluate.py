"""Evaluate a checkpoint on the official PlantVillage test split.

    python -m leafscan.evaluate models/efficientnet_b0/best.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader  # noqa: E402

from leafscan.data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    PlantVillageDataset,
    Sample,
    build_transforms,
    read_split,
)
from leafscan.model import get_device, load_checkpoint  # noqa: E402


@torch.no_grad()
def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    ys, probs = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized confusion matrix (test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_analysis(samples: list[Sample], classes: list[str], y_true: np.ndarray,
                   probs: np.ndarray, top: int = 10) -> tuple[dict, list[dict]]:
    """Most frequent confusions, confidence of right vs wrong predictions, list of errors."""
    y_pred, conf = probs.argmax(1), probs.max(1)
    correct = y_pred == y_true
    wrong = np.flatnonzero(~correct)
    pairs = Counter((classes[y_true[i]], classes[y_pred[i]]) for i in wrong)
    summary = {
        "n_errors": len(wrong),
        "top_confusions": [{"true": t, "pred": p, "count": n}
                           for (t, p), n in pairs.most_common(top)],
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_wrong": float(conf[wrong].mean()) if len(wrong) else None,
    }
    errors = [{"path": str(samples[i].path), "true": classes[y_true[i]],
               "pred": classes[y_pred[i]], "confidence": round(float(conf[i]), 4)}
              for i in wrong[np.argsort(-conf[wrong])]]
    return summary, errors


def evaluate_checkpoint(ckpt_path: Path, samples: list[Sample], out_dir: Path,
                        batch_size: int = 64, num_workers: int = 4) -> dict:
    device = get_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    classes = ckpt["classes"]
    ds = PlantVillageDataset(samples, classes, build_transforms(ckpt["img_size"], train=False))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    y_true, probs = predict_all(model, loader, device)
    y_pred = probs.argmax(1)
    labels = list(range(len(classes)))
    report = {
        "model": ckpt["model_name"],
        "n_test": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        # top-5 is meaningless (always 1.0) with 5 classes or fewer
        "top5_accuracy": (top_k_accuracy_score(y_true, probs, k=5, labels=labels)
                          if len(classes) > 5 else None),
        "errors": None,  # filled below
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                           output_dict=True, zero_division=0),
    }

    report["errors"], errors = error_analysis(samples, classes, y_true, probs)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_errors.json").write_text(json.dumps(errors, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    plot_confusion(confusion_matrix(y_true, y_pred, labels=labels), classes,
                   out_dir / "confusion_matrix.png")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = p.parse_args()

    report = evaluate_checkpoint(args.checkpoint, read_split(args.data_dir, "test"),
                                 args.checkpoint.parent)
    print(f"accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f} "
          f"top5={report['top5_accuracy']:.4f} (n={report['n_test']})")


if __name__ == "__main__":
    main()
