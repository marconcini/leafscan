"""Build the demo model: field fine-tuning + field calibration.

    python -m leafscan.deploy                      # -> models/leafscan_field/best.pt
                                                   #    reports/leafscan_field/deploy.json

ConvNeXt-T from the PlantVillage checkpoint, fine-tuned on 85% of PlantDoc train. The other
15% is a field calibration set, used only to fit the temperature and the abstention threshold.
PlantDoc test and PlantVillage test are used for reporting only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from leafscan.data import DEFAULT_DATA_DIR, load_splits
from leafscan.field import DEFAULT_PLANTDOC_DIR, field_metrics, load_plantdoc
from leafscan.field_finetune import (
    bootstrap_ci,
    init_model,
    predict_logits,
    stratified_fraction,
    train_steps,
)
from leafscan.model import get_device, save_checkpoint
from leafscan.reliability import ece, fit_temperature, lab_threshold, softmax

TARGET_ACCURACY = 0.90  # answers above the threshold should be right at least this often


def split_calibration(train_items: list[dict], calib_fraction: float, seed: int
                      ) -> tuple[list[dict], list[dict]]:
    kept = stratified_fraction(train_items, 1 - calib_fraction, seed)
    kept_paths = {str(i["path"]) for i in kept}
    return kept, [i for i in train_items if str(i["path"]) not in kept_paths]


def selective(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    answered = probs.max(1) >= threshold
    return {"coverage": float(answered.mean()),
            "accuracy_when_answering": float((probs.argmax(1) == y)[answered].mean())
            if answered.any() else None}


def build(pv_checkpoint: Path, out_dir: Path, plantdoc_dir: Path = DEFAULT_PLANTDOC_DIR,
          data_dir: Path = DEFAULT_DATA_DIR, steps: int = 400, calib_fraction: float = 0.15,
          seed: int = 0, device: torch.device | None = None, num_workers: int = 4) -> dict:
    device = device or get_device()
    torch.manual_seed(seed)
    items, _ = load_plantdoc(plantdoc_dir)
    train_items, calib_items = split_calibration(
        [i for i in items if i["split"] == "train"], calib_fraction, seed)
    test_items = [i for i in items if i["split"] == "test"]

    model, classes, img_size, name = init_model("plantvillage", pv_checkpoint, device)
    c2i = {c: i for i, c in enumerate(classes)}
    losses = train_steps(model, train_items, c2i, img_size, steps, 32, 1e-4, device, num_workers)

    yc, lc = predict_logits(model, calib_items, c2i, img_size, device, num_workers)
    t = fit_temperature(lc, yc)
    pc = softmax(lc, t)
    threshold = lab_threshold(pc.max(1), pc.argmax(1) == yc, TARGET_ACCURACY)

    yt, lt = predict_logits(model, test_items, c2i, img_size, device, num_workers)
    p1, pt = softmax(lt), softmax(lt, t)
    lo, hi = bootstrap_ci(pt.argmax(1) == yt)
    _, _, pv_test, _ = load_splits(data_dir)
    pv_items = [{"path": s.path, "label": s.label} for s in pv_test]
    yp, lp = predict_logits(model, pv_items, c2i, img_size, device, num_workers)

    save_checkpoint(out_dir / "best.pt", model, name, classes, img_size,
                    {"temperature": t, "threshold": threshold,
                     "threshold_target_accuracy": TARGET_ACCURACY,
                     "trained_on": "PlantVillage, then 85% of PlantDoc train"})
    return {
        "model": name, "steps": steps, "seed": seed,
        "n_train": len(train_items), "n_calibration": len(calib_items), "n_test": len(yt),
        "final_train_loss": float(np.mean(losses[-20:])),
        "temperature": t, "threshold": threshold,
        "plantdoc_test": {
            **{k: v for k, v in field_metrics(yt, pt, classes).items()
               if k not in ("per_class", "most_predicted", "top_confusions")},
            "accuracy_ci95": [lo, hi],
            "ece_uncalibrated": ece(p1, yt), "ece_calibrated": ece(pt, yt),
            "with_threshold": selective(pt, yt, threshold),
        },
        "plantdoc_calibration_set": {"with_threshold": selective(pc, yc, threshold)},
        "plantvillage_test": {"accuracy": float((lp.argmax(1) == yp).mean()),
                              "ece_calibrated": ece(softmax(lp, t), yp),
                              "with_threshold": selective(softmax(lp, t), yp, threshold)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pv-checkpoint", type=Path, default=Path("models/convnext_tiny/best.pt"))
    ap.add_argument("--name", default="leafscan_field")
    ap.add_argument("--device", help="e.g. cpu (default: cuda > mps > cpu)")
    args = ap.parse_args()

    res = build(args.pv_checkpoint, Path("models") / args.name,
                device=torch.device(args.device) if args.device else None)
    out = Path("reports") / args.name / "deploy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    pd, pv = res["plantdoc_test"], res["plantvillage_test"]
    print(f"T={res['temperature']:.2f} threshold={res['threshold']:.3f} | PlantDoc test acc "
          f"{pd['accuracy']:.3f} ECE {pd['ece_uncalibrated']:.3f}->{pd['ece_calibrated']:.3f} "
          f"thr {pd['with_threshold']} | PlantVillage test acc {pv['accuracy']:.4f}")


if __name__ == "__main__":
    main()
