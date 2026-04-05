"""Baseline model: Logistic Regression on frozen CLIP embeddings.

This provides a simple, non-deep-learning baseline to compare against
the full cross-attention model. It extracts CLIP image + text embeddings,
concatenates them, and fits a Logistic Regression classifier.

Usage:
    python -m src.baseline
    python -m src.baseline --config config/config.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def extract_clip_features(
    records: list[dict],
    data_dir: str,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract concatenated [image_emb, text_emb] from CLIP for all samples.

    Args:
        records: List of dicts with 'img', 'text', and optionally 'label'.
        data_dir: Root data directory.
        clip_model: Frozen CLIP model.
        processor: CLIP image processor.
        tokenizer: CLIP tokenizer.
        device: Torch device.
        batch_size: Batch size for encoding.

    Returns:
        Tuple of (features [N, 2*clip_dim], labels [N]).
    """
    all_features = []
    all_labels = []
    data_path = Path(data_dir)

    for i in tqdm(range(0, len(records), batch_size), desc="Extracting CLIP features"):
        batch = records[i : i + batch_size]

        images = []
        texts = []
        labels = []
        for r in batch:
            img_path = data_path / r["img"]
            images.append(Image.open(img_path).convert("RGB"))
            texts.append(r.get("text", "") or " ")
            labels.append(r.get("label", -1))

        # Image encoding
        pixel_values = processor(
            images=images, return_tensors="pt"
        )["pixel_values"].to(device)

        # Text encoding
        text_inputs = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=77, return_tensors="pt",
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

            # Concatenate image + text embeddings
            combined = torch.cat([img_emb, txt_emb], dim=1).cpu().numpy()

        all_features.append(combined)
        all_labels.extend(labels)

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)
    return features, labels


def main():
    parser = argparse.ArgumentParser(description="Baseline: Logistic Regression on CLIP features")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load CLIP
    clip_name = cfg["model"]["clip_model"]
    logger.info("Loading CLIP: %s", clip_name)
    processor = CLIPProcessor.from_pretrained(clip_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Load data
    data_dir = cfg["data"]["data_dir"]
    data_path = Path(data_dir)

    def load_jsonl(path):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    train_records = load_jsonl(data_path / cfg["data"]["train_file"])
    dev_records = load_jsonl(data_path / cfg["data"]["dev_file"])

    logger.info("Train: %d samples, Dev: %d samples", len(train_records), len(dev_records))

    # Extract features
    logger.info("Extracting CLIP features for train set...")
    X_train, y_train = extract_clip_features(
        train_records, data_dir, clip_model, processor, tokenizer, device,
        batch_size=cfg["training"]["batch_size"],
    )
    logger.info("Train features: %s", X_train.shape)

    logger.info("Extracting CLIP features for dev set...")
    X_dev, y_dev = extract_clip_features(
        dev_records, data_dir, clip_model, processor, tokenizer, device,
        batch_size=cfg["training"]["batch_size"],
    )
    logger.info("Dev features: %s", X_dev.shape)

    # Fit Logistic Regression
    logger.info("Training Logistic Regression baseline...")
    start = time.time()
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info("Logistic Regression trained in %.1fs", train_time)

    # Evaluate on train
    train_probs = clf.predict_proba(X_train)[:, 1]
    train_preds = clf.predict(X_train)
    train_auroc = roc_auc_score(y_train, train_probs)
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds, average="macro")

    # Evaluate on dev
    dev_probs = clf.predict_proba(X_dev)[:, 1]
    dev_preds = clf.predict(X_dev)
    dev_auroc = roc_auc_score(y_dev, dev_probs)
    dev_acc = accuracy_score(y_dev, dev_preds)
    dev_f1 = f1_score(y_dev, dev_preds, average="macro")

    # Report
    logger.info("=" * 60)
    logger.info("BASELINE RESULTS: Logistic Regression on CLIP Embeddings")
    logger.info("=" * 60)
    logger.info("  Feature dim:    %d (image %d + text %d)",
                X_train.shape[1], X_train.shape[1] // 2, X_train.shape[1] // 2)
    logger.info("")
    logger.info("  TRAIN:  AUROC=%.4f  Acc=%.4f  F1=%.4f", train_auroc, train_acc, train_f1)
    logger.info("  DEV:    AUROC=%.4f  Acc=%.4f  F1=%.4f", dev_auroc, dev_acc, dev_f1)
    logger.info("")
    logger.info("Dev Classification Report:")
    logger.info("\n%s", classification_report(
        y_dev, dev_preds, target_names=["not_hateful", "hateful"], zero_division=0
    ))

    # Save confusion matrix
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_dev, dev_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["not_hateful", "hateful"])
    disp.plot(ax=ax, cmap="Oranges")
    ax.set_title("Baseline: Logistic Regression Confusion Matrix (Dev)", fontsize=12, fontweight="bold")
    fig.savefig(out_dir / "baseline_confusion_matrix.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_dir / "baseline_confusion_matrix.png")

    # Save results summary
    results_path = out_dir / "baseline_results.txt"
    with open(results_path, "w") as f:
        f.write("BASELINE: Logistic Regression on Frozen CLIP Embeddings\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"CLIP Model: {clip_name}\n")
        f.write(f"Feature dim: {X_train.shape[1]} (image {X_train.shape[1]//2} + text {X_train.shape[1]//2})\n")
        f.write(f"Classifier: LogisticRegression(C=1.0, class_weight='balanced')\n")
        f.write(f"Training time: {train_time:.1f}s\n\n")
        f.write(f"TRAIN:  AUROC={train_auroc:.4f}  Acc={train_acc:.4f}  F1={train_f1:.4f}\n")
        f.write(f"DEV:    AUROC={dev_auroc:.4f}  Acc={dev_acc:.4f}  F1={dev_f1:.4f}\n\n")
        f.write("Dev Classification Report:\n")
        f.write(classification_report(
            y_dev, dev_preds, target_names=["not_hateful", "hateful"], zero_division=0
        ))
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
