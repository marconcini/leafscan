"""Hugging Face Space entry point: the LeafScan Gradio demo, weights from the Hub."""

import os

from leafscan.app import build_app
from leafscan.predictor import Predictor

MODEL = os.environ.get("LEAFSCAN_MODEL", "Marconcini/leafscan-convnext-tiny")

demo = build_app(Predictor(MODEL))

if __name__ == "__main__":
    demo.launch()
