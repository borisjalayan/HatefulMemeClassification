"""Gradio web interface for the Hateful Meme Classifier."""

import logging
import sys
from pathlib import Path

import gradio as gr

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import MemePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/config.yaml"
CHECKPOINT_PATH = "checkpoints/best_model.pt"

predictor: MemePredictor | None = None


def get_predictor() -> MemePredictor:
    """Lazy-load the predictor singleton."""
    global predictor
    if predictor is None:
        predictor = MemePredictor(CONFIG_PATH, CHECKPOINT_PATH)
    return predictor


def classify_meme(image_path: str) -> tuple[str, str, str, str]:
    """Classify an uploaded meme image.

    Args:
        image_path: Path to the uploaded image (provided by Gradio).

    Returns:
        Tuple of (extracted_text, label_html, confidence_text, confidence_bar_html).
    """
    if image_path is None:
        return "No image uploaded.", "", "", ""

    pred = get_predictor()
    result = pred.predict(image_path)

    extracted_text = result["extracted_text"]
    label = result["label"]
    confidence = result["confidence"]
    hateful_prob = result["hateful_probability"]

    # Color-coded label
    if label == "hateful":
        label_html = (
            f'<div style="background-color: #e74c3c; color: white; '
            f'padding: 20px; border-radius: 10px; text-align: center; '
            f'font-size: 28px; font-weight: bold;">'
            f'⚠️ HATEFUL</div>'
        )
    else:
        label_html = (
            f'<div style="background-color: #27ae60; color: white; '
            f'padding: 20px; border-radius: 10px; text-align: center; '
            f'font-size: 28px; font-weight: bold;">'
            f'✅ NOT HATEFUL</div>'
        )

    confidence_text = f"Confidence: {confidence:.1%}"

    # Confidence bar
    bar_color = "#e74c3c" if label == "hateful" else "#27ae60"
    bar_pct = confidence * 100
    confidence_bar_html = (
        f'<div style="background: #eee; border-radius: 8px; overflow: hidden; '
        f'height: 30px; margin-top: 10px;">'
        f'<div style="background: {bar_color}; width: {bar_pct:.0f}%; '
        f'height: 100%; border-radius: 8px; display: flex; align-items: center; '
        f'justify-content: center; color: white; font-weight: bold;">'
        f'{confidence:.1%}</div></div>'
        f'<div style="margin-top: 5px; text-align: center; color: #666;">'
        f'Hateful probability: {hateful_prob:.3f}</div>'
    )

    return extracted_text, label_html, confidence_text, confidence_bar_html


def build_app() -> gr.Blocks:
    """Construct the Gradio interface."""
    with gr.Blocks(
        title="Hateful Meme Classifier",
    ) as app:
        gr.Markdown(
            "# Hateful Meme Classifier\n"
            "Upload a meme image to classify whether it contains hateful content.\n"
            "The model uses CLIP vision + text encoders with cross-attention fusion."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Meme",
                    type="filepath",
                    height=400,
                )
                classify_btn = gr.Button("Classify", variant="primary", size="lg")

            with gr.Column(scale=1):
                label_output = gr.HTML(label="Prediction")
                confidence_bar = gr.HTML(label="Confidence")
                confidence_text = gr.Textbox(label="Confidence Score", interactive=False)
                extracted_text_output = gr.Textbox(
                    label="Extracted Text (OCR)",
                    interactive=False,
                    lines=3,
                )

        classify_btn.click(
            fn=classify_meme,
            inputs=[image_input],
            outputs=[extracted_text_output, label_output, confidence_text, confidence_bar],
        )

    return app


if __name__ == "__main__":
    # Eagerly load model + OCR so first request isn't slow
    logger.info("Pre-loading model and OCR engine...")
    get_predictor()
    try:
        from src.ocr import _get_doctr_model, _build_jsonl_index
        _build_jsonl_index()
        _get_doctr_model()
    except Exception:
        pass
    logger.info("Ready — launching Gradio")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
