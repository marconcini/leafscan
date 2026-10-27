"""LeafScan demo (Gradio).

    pip install -e ".[app]"
    python -m leafscan.app                         # http://127.0.0.1:7860
    LEAFSCAN_CHECKPOINT=path/to/best.pt python -m leafscan.app
"""

from __future__ import annotations

import os

import gradio as gr

from leafscan.data import pretty_label
from leafscan.predictor import Predictor

CHECKPOINT = os.environ.get("LEAFSCAN_CHECKPOINT", "models/leafscan_field/best.pt")

ABOUT = """
**What this is.** A research demo: a ConvNeXt-T trained on lab photos (PlantVillage), then
fine-tuned on web and field photos (PlantDoc), for **38 crop / disease classes** of 14 crops.

**How much to trust it.** About **75%** of field photos are classified correctly. When the
model is not confident enough to be right at least ~90% of the time (threshold fitted on field
photos), it says so instead of answering. It only knows the 38 classes it was trained on, and
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
    build_app(Predictor(CHECKPOINT)).launch()


if __name__ == "__main__":
    main()
