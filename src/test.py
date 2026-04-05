"""Generate predictions for unlabelled test data and save to CSV.

Reads test.jsonl (which has no labels), runs each sample through the
trained model, and produces a submission-ready CSV:

    outputs/test_predictions.csv

Columns: id, label, hateful_probability, text

Usage:
    python src/test.py
    python src/test.py --config config/config.yaml --checkpoint checkpoints/best_model.pt
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import HatefulMemeClassifier
from src.train import compute_hateful_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test set predictions")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Path to test JSONL (default: from config)")
    parser.add_argument("--output", type=str, default="outputs/test_predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data"]["data_dir"])
    test_path = Path(args.test_file) if args.test_file else data_dir / cfg["data"]["test_file"]
    threshold = cfg["inference"]["threshold"]

    if not test_path.exists():
        logger.error("Test file not found: %s", test_path)
        sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ── Load CLIP ────────────────────────────────────────────────────────
    clip_name = cfg["model"]["clip_model"]
    logger.info("Loading CLIP: %s", clip_name)
    processor = CLIPProcessor.from_pretrained(clip_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ── Load classifier ──────────────────────────────────────────────────
    hateful_emb = compute_hateful_embedding(clip_model, clip_name, device)
    model = HatefulMemeClassifier(
        clip_dim=cfg["model"]["clip_dim"],
        hidden=cfg["model"]["hidden"],
        num_heads=cfg["model"]["num_heads"],
        hateful_emb=hateful_emb,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s (epoch %d)",
                args.checkpoint, ckpt.get("epoch", -1))

    # ── Load test records ────────────────────────────────────────────────
    records = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d test samples from %s", len(records), test_path)

    # ── Batch inference ──────────────────────────────────────────────────
    results = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(records), batch_size), desc="Predicting"):
        batch = records[i : i + batch_size]

        # Prepare images
        images = []
        texts = []
        ids = []
        for r in batch:
            img_path = data_dir / r["img"]
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            # Use text from JSONL (ground-truth text for test set)
            text = r.get("text", "").strip() or " "
            texts.append(text)
            ids.append(r["id"])

        # CLIP image encoding
        pixel_values = processor(
            images=images, return_tensors="pt"
        )["pixel_values"].to(device)

        # CLIP text encoding
        text_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        with torch.no_grad():
            vis_out = clip_model.vision_model(pixel_values=pixel_values)
            img_emb = clip_model.visual_projection(vis_out.pooler_output)
            txt_out = clip_model.text_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            txt_emb = clip_model.text_projection(txt_out.pooler_output)

            logits = model(img_emb, txt_emb)
            probs = torch.sigmoid(logits).cpu().tolist()

        for sample_id, text, prob in zip(ids, texts, probs):
            label = 1 if prob >= threshold else 0
            results.append({
                "id": sample_id,
                "label": label,
                "hateful_probability": round(prob, 6),
                "text": text,
            })

    # ── Save CSV ─────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "hateful_probability", "text"])
        writer.writeheader()
        writer.writerows(results)

    # ── Summary stats ────────────────────────────────────────────────────
    total = len(results)
    hateful = sum(1 for r in results if r["label"] == 1)
    not_hateful = total - hateful
    avg_prob = sum(r["hateful_probability"] for r in results) / total if total else 0

    logger.info("=" * 60)
    logger.info("TEST PREDICTIONS COMPLETE")
    logger.info("=" * 60)
    logger.info("  Total samples:    %d", total)
    logger.info("  Predicted hateful: %d (%.1f%%)", hateful, hateful / total * 100)
    logger.info("  Predicted not:     %d (%.1f%%)", not_hateful, not_hateful / total * 100)
    logger.info("  Avg hateful prob:  %.4f", avg_prob)
    logger.info("  Threshold:         %.2f", threshold)
    logger.info("  Saved to:          %s", out_path.resolve())


if __name__ == "__main__":
    main()
