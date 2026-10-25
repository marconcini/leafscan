import json
from collections import Counter

import numpy as np
from PIL import Image

from leafscan.field_finetune import bootstrap_ci, run, stratified_fraction, summarize
from leafscan.model import create_model, save_checkpoint


def test_stratified_fraction_keeps_every_class():
    items = [{"label": "a"}] * 40 + [{"label": "b"}] * 10 + [{"label": "c"}] * 1
    sub = stratified_fraction(items, 0.1, seed=0)
    assert Counter(i["label"] for i in sub) == {"a": 4, "b": 1, "c": 1}
    assert stratified_fraction(items, 1.0, seed=0) is not items  # a copy, same size
    assert len(stratified_fraction(items, 1.0, seed=0)) == len(items)


def test_bootstrap_ci_brackets_the_accuracy():
    correct = np.array([1] * 60 + [0] * 40, dtype=bool)
    lo, hi = bootstrap_ci(correct)
    assert lo < 0.6 < hi and 0.45 < lo and hi < 0.75


def test_run_and_summarize(tmp_path):
    plantdoc = tmp_path / "plantdoc"
    for split, n in (("train", 6), ("test", 2)):
        for k, folder in enumerate(("Apple Scab Leaf", "Apple leaf")):
            d = plantdoc / split / folder
            d.mkdir(parents=True)
            for i in range(n):
                shade = 30 if split == "train" else 200  # no train/test duplicates
                Image.new("RGB", (40, 40), (10 * i, 100 + 50 * k, shade)).save(d / f"{i}.png")
    classes = ["Apple___Apple_scab", "Apple___healthy", "Tomato___healthy"]
    ckpt = tmp_path / "pv" / "best.pt"
    save_checkpoint(ckpt, create_model("resnet18", 3, pretrained=False), "resnet18", classes, 32)

    res = run("plantvillage", 0.5, 0, ckpt, plantdoc, steps=3, batch_size=4, num_workers=0)
    assert res["n_train"] == 6 and res["n"] == 4 and res["steps"] == 3
    lo, hi = res["accuracy_ci95"]
    assert lo <= res["accuracy"] <= hi

    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    for seed, acc in ((0, 0.5), (1, 0.7)):
        (report_dir / f"plantvillage_frac0.50_seed{seed}.json").write_text(json.dumps(
            {**res, "seed": seed, "accuracy": acc}))
    rows = summarize(report_dir)
    assert len(rows) == 1 and rows[0]["seeds"] == 2
    assert abs(rows[0]["accuracy_mean"] - 0.6) < 1e-9
