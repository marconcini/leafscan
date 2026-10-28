"""LeafScan demo (Gradio).

    pip install -e ".[app]"
    python -m leafscan.app                         # http://127.0.0.1:7860
    LEAFSCAN_MODEL=Marconcini/leafscan-convnext-tiny python -m leafscan.app   # from the Hub
"""

from __future__ import annotations

import os

import gradio as gr

from leafscan.data import pretty_label
from leafscan.predictor import Predictor

# a .pt checkpoint, an exported folder, or a Hugging Face Hub repo id
MODEL = os.environ.get("LEAFSCAN_MODEL", "models/leafscan_field/best.pt")

ABOUT = """
**What this is.** A research demo: a ConvNeXt-T trained on lab photos (PlantVillage), then
fine-tuned on web and field photos (PlantDoc), for **38 crop / disease classes** of 14 crops.

**How much to trust it.** On held-out field photos it is right **72%** of the time. When it is
not confident enough (threshold fitted on field photos) it says "Not sure" instead of answering:
it answers about two thirds of field photos, and **~87%** of those answers are right.
It only knows the 38 classes it was trained on, and
it is **not** a substitute for an agronomist.

Details, experiments and limitations: [github.com/marconcini/leafscan](https://github.com/marconcini/leafscan)
"""


def build_app(predictor: Predictor) -> gr.Blocks:
    def run(image):
        if image is None:
            return {}, "Upload a photo of a leaf.", None
        pred = predictor(image)
        scores = {pretty_label(c): p for c, p in pred.top}
        return scores, pred.verdict, predictor.overlay(image, pred.heatmap)

    with gr.Blocks(title="LeafScan") as demo:
        gr.Markdown("# 🌿 LeafScan — plant disease from a leaf photo")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Leaf photo")
                button = gr.Button("Diagnose", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                scores = gr.Label(num_top_classes=3, label="Top 3 (calibrated)")
                heat = gr.Image(label="Where the model looked (Grad-CAM)")
        gr.Markdown(ABOUT)
        button.click(run, inputs=image, outputs=[scores, verdict, heat])
        image.upload(run, inputs=image, outputs=[scores, verdict, heat])
    return demo


def main() -> None:
    build_app(Predictor(MODEL)).launch()


if __name__ == "__main__":
    main()
