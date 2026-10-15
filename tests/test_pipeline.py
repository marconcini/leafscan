import json

import torch

from leafscan.model import create_model, load_checkpoint
from leafscan.train import DEFAULTS, train


def test_model_output_shape():
    model = create_model("resnet18", num_classes=38, pretrained=False)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 38)


def test_end_to_end_training_and_evaluation(fake_data_dir, tmp_path):
    cfg = {**DEFAULTS, "model": "resnet18", "pretrained": False, "img_size": 32,
           "epochs": 1, "batch_size": 8, "num_workers": 0, "val_fraction": 0.2}
    out = tmp_path / "run"
    ckpt = train(cfg, fake_data_dir, out)

    model, meta = load_checkpoint(ckpt, torch.device("cpu"))
    assert meta["classes"] == ["Apple___Black_rot", "Apple___healthy", "Tomato___Early_blight"]
    metrics = json.loads((out / "test_metrics.json").read_text())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert (out / "confusion_matrix.png").exists()
    errors = json.loads((out / "test_errors.json").read_text())
    assert len(errors) == metrics["errors"]["n_errors"]
    assert len(json.loads((out / "history.json").read_text())) == 1


def test_error_analysis_counts_confusions():
    import numpy as np

    from leafscan.data import Sample
    from leafscan.evaluate import error_analysis

    classes = ["a", "b", "c"]
    samples = [Sample(f"img{i}.jpg", "x") for i in range(4)]
    y_true = np.array([0, 0, 1, 2])
    probs = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1], [0.6, 0.0, 0.4]])
    summary, errors = error_analysis(samples, classes, y_true, probs)
    assert summary["n_errors"] == 2
    assert {(c["true"], c["pred"]) for c in summary["top_confusions"]} == {("a", "b"), ("c", "a")}
    assert [e["confidence"] for e in errors] == [0.7, 0.6]  # most confident errors first
