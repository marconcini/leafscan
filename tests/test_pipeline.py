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
    assert len(json.loads((out / "history.json").read_text())) == 1
