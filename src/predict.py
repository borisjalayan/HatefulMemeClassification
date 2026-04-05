"""Single-image inference pipeline for the Hateful Meme Classifier."""

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from src.model import HatefulMemeClassifier
from src.ocr import extract_text
from src.train import compute_hateful_embedding

logger = logging.getLogger(__name__)


class MemePredictor:
    """End-to-end inference pipeline: image → OCR → CLIP encode → classify."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        checkpoint_path: str = "checkpoints/best_model.pt",
    ) -> None:
        """Initialise the predictor.

        Args:
            config_path: Path to config YAML.
            checkpoint_path: Path to trained model checkpoint.
        """
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("Predictor using device: %s", self.device)

        clip_name = self.cfg["model"]["clip_model"]
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)

        # Frozen CLIP encoder
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Hateful query embedding
        hateful_emb = compute_hateful_embedding(self.clip_model, clip_name, self.device)

        # Classifier
        self.model = HatefulMemeClassifier(
            clip_dim=self.cfg["model"]["clip_dim"],
            hidden=self.cfg["model"]["hidden"],
            num_heads=self.cfg["model"]["num_heads"],
            hateful_emb=hateful_emb,
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.threshold = self.cfg["inference"]["threshold"]
        self.use_ocr = self.cfg["inference"]["use_ocr"]
        logger.info("Predictor ready (threshold=%.2f, ocr=%s)", self.threshold, self.use_ocr)

    def predict(self, image_path: str, text: str | None = None) -> dict[str, Any]:
        """Run inference on a single meme image.

        Args:
            image_path: Path to the meme image.
            text: Optional pre-extracted text. If None and use_ocr is True,
                  OCR will be run on the image.

        Returns:
            Dictionary with keys: label, confidence, extracted_text.
        """
        # Text extraction
        if text is None:
            extracted_text = extract_text(image_path) if self.use_ocr else ""
        else:
            extracted_text = text

        if not extracted_text:
            extracted_text = " "
            logger.warning("No text extracted — using placeholder")

        # Image preprocessing
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(self.device)

        # Text tokenization
        text_inputs = self.tokenizer(
            extracted_text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(self.device)
        attention_mask = text_inputs["attention_mask"].to(self.device)

        # CLIP encoding
        with torch.no_grad():
            vis_out = self.clip_model.vision_model(pixel_values=pixel_values)
            img_emb = self.clip_model.visual_projection(vis_out.pooler_output)
            txt_out = self.clip_model.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            txt_emb = self.clip_model.text_projection(txt_out.pooler_output)

            logits = self.model(img_emb, txt_emb)
            hateful_prob = torch.sigmoid(logits).item()

        is_hateful = hateful_prob >= self.threshold

        result = {
            "label": "hateful" if is_hateful else "not hateful",
            "confidence": hateful_prob if is_hateful else 1.0 - hateful_prob,
            "hateful_probability": hateful_prob,
            "extracted_text": extracted_text,
        }
        logger.info("Prediction: %s (conf=%.3f)", result["label"], result["confidence"])
        return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    import argparse
    parser = argparse.ArgumentParser(description="Predict on a single meme image")
    parser.add_argument("image", type=str, help="Path to meme image")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--text", type=str, default=None, help="Optional pre-extracted text")
    args = parser.parse_args()

    predictor = MemePredictor(args.config, args.checkpoint)
    result = predictor.predict(args.image, args.text)
    for k, v in result.items():
        print(f"{k}: {v}")
