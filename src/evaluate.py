"""Evaluation and threshold tuning for the Hateful Meme Classifier."""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from src.dataset import HatefulMemesDataset
from src.model import HatefulMemeClassifier
from src.train import compute_hateful_embedding, encode_batch

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> HatefulMemeClassifier:
    """Load a trained model from checkpoint.

    Args:
        config: Model configuration dictionary.
        checkpoint_path: Path to saved checkpoint.
        device: Device to load model onto.

    Returns:
        The loaded model in eval mode.
    """
    clip_model_name = config["model"]["clip_model"]
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    hateful_emb = compute_hateful_embedding(clip_model, clip_model_name, device)

    model = HatefulMemeClassifier(
        clip_dim=config["model"]["clip_dim"],
        hidden=config["model"]["hidden"],
        num_heads=config["model"]["num_heads"],
        hateful_emb=hateful_emb,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s (epoch %d, AUROC %.4f)",
                checkpoint_path, ckpt.get("epoch", -1), ckpt.get("dev_auroc", ckpt.get("dev_auroc", -1)))
    return model, clip_model


def collect_predictions(
    model: HatefulMemeClassifier,
    clip_model: CLIPModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Run inference and collect labels + predicted probabilities.

    Args:
        model: Trained model.
        clip_model: Frozen CLIP encoder.
        loader: DataLoader for evaluation split.
        device: Torch device.

    Returns:
        Tuple of (labels, probabilities) for the positive class.
    """
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            img_emb, txt_emb = encode_batch(clip_model, pixel_values, input_ids, attention_mask)
            logits = model(img_emb, txt_emb)
            probs = torch.sigmoid(logits).cpu().tolist()

            all_labels.extend(labels.tolist())
            all_probs.extend(probs)

    return all_labels, all_probs


def find_optimal_threshold(
    labels: list[int],
    probs: list[float],
    low: float = 0.1,
    high: float = 0.9,
    step: float = 0.01,
) -> tuple[float, float]:
    """Sweep thresholds and pick the one that maximises macro F1.

    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities for positive class.
        low: Lower bound of sweep range.
        high: Upper bound of sweep range.
        step: Step size.

    Returns:
        Tuple of (best_threshold, best_f1).
    """
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(low, high + step, step):
        preds = [1 if p >= t else 0 for p in probs]
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
    logger.info("Optimal threshold: %.2f  (macro F1: %.4f)", best_thresh, best_f1)
    return best_thresh, best_f1


def report_metrics(
    labels: list[int],
    probs: list[float],
    threshold: float,
    save_dir: str = "logs",
) -> dict[str, float]:
    """Compute and log all evaluation metrics.

    Args:
        labels: Ground truth labels.
        probs: Predicted probabilities for positive class.
        threshold: Classification threshold.
        save_dir: Directory to save confusion matrix image.

    Returns:
        Dictionary of computed metrics.
    """
    preds = [1 if p >= threshold else 0 for p in probs]

    auroc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    logger.info("=" * 50)
    logger.info("Evaluation Results (threshold=%.2f)", threshold)
    logger.info("=" * 50)
    logger.info("AUROC     : %.4f", auroc)
    logger.info("Accuracy  : %.4f", acc)
    logger.info("Precision : %.4f (macro)", prec)
    logger.info("Recall    : %.4f (macro)", rec)
    logger.info("F1        : %.4f (macro)", f1)
    logger.info("-" * 50)
    logger.info("\n%s", classification_report(
        labels, preds, target_names=["not_hateful", "hateful"], zero_division=0,
    ))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["not_hateful", "hateful"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    cm_path = save_path / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", cm_path)

    return {"auroc": auroc, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate(
    config_path: str = "config/config.yaml",
    split: str = "dev",
    checkpoint_path: str = "checkpoints/best_model.pt",
) -> None:
    """Main evaluation entry point.

    Args:
        config_path: Path to config YAML.
        split: Which split to evaluate on (dev or test).
        checkpoint_path: Path to model checkpoint.
    """
    setup_logging()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Using device: %s", device)

    # Dataset
    data_dir = cfg["data"]["data_dir"]
    split_file = cfg["data"]["dev_file"] if split == "dev" else cfg["data"]["test_file"]
    ds = HatefulMemesDataset(
        jsonl_path=str(Path(data_dir) / split_file),
        data_dir=data_dir,
        clip_model_name=cfg["model"]["clip_model"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model, clip_model = load_model(cfg, checkpoint_path, device)
    labels, probs = collect_predictions(model, clip_model, loader, device)

    # Check if labels are valid (test split has label=-1, i.e. no ground truth)
    valid_labels = [l for l in labels if l in (0, 1)]
    if len(valid_labels) < len(labels):
        n_invalid = len(labels) - len(valid_labels)
        logger.warning(
            "%d / %d samples have no ground-truth label (label=-1). "
            "Cannot compute metrics on the test split — use --split dev instead, "
            "or use test.py to generate predictions.",
            n_invalid, len(labels),
        )
        if not valid_labels:
            logger.error("No labelled samples found. Aborting evaluation.")
            return
        # Filter to only labelled samples
        filtered = [(l, p) for l, p in zip(labels, probs) if l in (0, 1)]
        labels = [l for l, _ in filtered]
        probs = [p for _, p in filtered]
        logger.info("Continuing with %d labelled samples.", len(labels))

    # Threshold tuning on dev set
    best_thresh, _ = find_optimal_threshold(labels, probs)
    report_metrics(labels, probs, threshold=best_thresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hateful Meme Classifier")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()
    evaluate(args.config, args.split, args.checkpoint)
