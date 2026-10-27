import numpy as np
import pytest
import torch
from PIL import Image

from leafscan.model import create_model, save_checkpoint
from leafscan.predictor import Prediction, Predictor

CLASSES = ["Apple___Apple_scab", "Apple___healthy", "Tomato___Late_blight"]


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "best.pt"
    torch.manual_seed(0)
    save_checkpoint(path, create_model("resnet18", 3, pretrained=False), "resnet18", CLASSES,
                    img_size=32, extra={"temperature": 2.0, "threshold": 0.5})
    return path


def test_predictor_returns_calibrated_sorted_top_k(ckpt):
    pred = Predictor(ckpt)(Image.new("RGB", (50, 40), (60, 150, 60)), k=3)
    probs = [p for _, p in pred.top]
    assert probs == sorted(probs, reverse=True) and abs(sum(probs) - 1) < 1e-5
    assert {c for c, _ in pred.top} == set(CLASSES)
    assert pred.confident == (probs[0] >= 0.5)
    assert pred.heatmap.shape == (32, 32)


def test_overlay_has_model_input_size(ckpt):
    p = Predictor(ckpt)
    img = Image.new("RGB", (80, 60), (60, 150, 60))
    out = p.overlay(img, np.zeros((32, 32)))
    assert out.size == (32, 32)


def test_verdict_abstains_below_threshold():
    sure = Prediction([("Apple___healthy", 0.95)], True, 0.8)
    unsure = Prediction([("Apple___healthy", 0.40)], False, 0.8)
    assert "Apple — healthy" in sure.verdict and "Not sure" not in sure.verdict
    assert unsure.verdict.startswith("**Not sure.**") and "80%" in unsure.verdict


def test_app_builds(ckpt):
    gr = pytest.importorskip("gradio")
    from leafscan.app import build_app

    assert isinstance(build_app(Predictor(ckpt)), gr.Blocks)
