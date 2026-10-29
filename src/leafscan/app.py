"""LeafScan demo (Gradio).

    pip install -e ".[app]"
    python -m leafscan.app                         # http://127.0.0.1:7860
    LEAFSCAN_MODEL=Marconcini/leafscan-convnext-tiny python -m leafscan.app   # from the Hub

    pip install -e ".[app,rag]"
    LEAFSCAN_ASSISTANT=1 python -m leafscan.app    # + the question-answering assistant
                                                   # (downloads ~10 GB of models on first run)
"""

from __future__ import annotations

import os

import gradio as gr

from leafscan.data import pretty_label
from leafscan.predictor import Predictor

# a .pt checkpoint, an exported folder, or a Hugging Face Hub repo id
MODEL = os.environ.get("LEAFSCAN_MODEL", "models/leafscan_field/best.pt")
ASSISTANT = os.environ.get("LEAFSCAN_ASSISTANT") == "1"

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

ABOUT_ASSISTANT = """
**The assistant** answers from 28 Wikipedia articles only, with a local LLM (Qwen3-4B), and
cites its sources. It says "I don't know" when they do not cover the question. A sentence its
cited source does not support (checked by an NLI model) is flagged with ⚠️, but the check
misses some inventions: on our test questions, ~12% of answer sentences were embellishments.
Read the cited sources before acting.
Your question is searched **as written**: name the crop and the disease (the suggested
questions do), since a wrong diagnosis must not steer the search.
"""


def load_assistant():
    from leafscan.rag.assistant import Assistant
    from leafscan.rag.attribution import NLI
    from leafscan.rag.corpus import chunk_pages, load_pages
    from leafscan.rag.generate import RETRIEVER, Generator
    from leafscan.rag.retrieval import Dense

    chunks = chunk_pages(load_pages(), 200)
    return Assistant(chunks, Dense(chunks, RETRIEVER), Generator(), NLI())


def build_app(predictor: Predictor, assistant=None) -> gr.Blocks:
    from leafscan.rag.assistant import render, suggested_questions

    def run(image):
        if image is None:
            return {}, "Upload a photo of a leaf.", None, gr.update(choices=[], value=None)
        pred = predictor(image)
        scores = {pretty_label(c): p for c, p in pred.top}
        suggestions = suggested_questions(pred.top, pred.confident)
        return (scores, pred.verdict, predictor.overlay(image, pred.heatmap),
                gr.update(choices=suggestions, value=None))

    def ask(question):
        if not question or not question.strip():
            return "Type a question, or pick a suggested one."
        return render(assistant(question.strip()))

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
        suggestions = gr.Radio([], label="Suggested questions", visible=assistant is not None)
        if assistant is not None:
            gr.Markdown("## Ask about the disease")
            with gr.Row():
                question = gr.Textbox(label="Your question", scale=4,
                                      placeholder="e.g. How can I manage early blight on tomato?")
                ask_button = gr.Button("Ask", variant="primary", scale=1)
            answer = gr.Markdown()
            suggestions.select(lambda q: q, inputs=suggestions, outputs=question)
            ask_button.click(ask, inputs=question, outputs=answer)
            question.submit(ask, inputs=question, outputs=answer)
            gr.Markdown(ABOUT_ASSISTANT)
        gr.Markdown(ABOUT)
        outputs = [scores, verdict, heat, suggestions]
        button.click(run, inputs=image, outputs=outputs)
        image.upload(run, inputs=image, outputs=outputs)
    return demo


def main() -> None:
    build_app(Predictor(MODEL), load_assistant() if ASSISTANT else None).launch()


if __name__ == "__main__":
    main()
