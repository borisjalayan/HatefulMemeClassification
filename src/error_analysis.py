"""Error analysis: find and visualise misclassified memes.

Runs the trained model on the dev set, identifies misclassified samples,
and produces visualisations showing failure cases with explanations.

Usage:
    python -m src.error_analysis
    python -m src.error_analysis --config config/config.yaml --top-k 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from src.model import HatefulMemeClassifier
from src.train import compute_hateful_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def run_inference(
    records: list[dict],
    data_dir: str,
    model: HatefulMemeClassifier,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    tokenizer: CLIPTokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> list[dict]:
    """Run inference on all records and return enriched results.

    Returns list of dicts with original fields + 'prob', 'pred', 'correct'.
    """
    results = []
    data_path = Path(data_dir)

    for i in tqdm(range(0, len(records), batch_size), desc="Inference"):
        batch = records[i : i + batch_size]

        images = []
        texts = []
        for r in batch:
            images.append(Image.open(data_path / r["img"]).convert("RGB"))
            texts.append(r.get("text", "") or " ")

        pixel_values = processor(
            images=images, return_tensors="pt"
        )["pixel_values"].to(device)

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
            logits = model(img_emb, txt_emb)
            probs = torch.sigmoid(logits).cpu().tolist()

        for r, prob in zip(batch, probs):
            pred = 1 if prob >= 0.5 else 0
            label = r.get("label", -1)
            results.append({
                **r,
                "prob": prob,
                "pred": pred,
                "correct": pred == label,
            })

    return results


def plot_failure_grid(
    failures: list[dict],
    data_dir: str,
    title: str,
    save_path: str,
    n_cols: int = 4,
) -> None:
    """Plot a grid of misclassified memes with details."""
    n = len(failures)
    if n == 0:
        return

    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, r in enumerate(failures):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        img_path = Path(data_dir) / r["img"]
        if img_path.exists():
            im = Image.open(img_path)
            ax.imshow(im)

        true_label = "HATEFUL" if r["label"] == 1 else "NOT HATEFUL"
        pred_label = "HATEFUL" if r["pred"] == 1 else "NOT HATEFUL"

        true_color = "#e74c3c" if r["label"] == 1 else "#27ae60"
        pred_color = "#e74c3c" if r["pred"] == 1 else "#27ae60"

        ax.set_title(
            f"ID: {r['id']}\n"
            f"True: {true_label} | Pred: {pred_label}\n"
            f"P(hateful) = {r['prob']:.3f}",
            fontsize=9, fontweight="bold",
        )

        # Add colored border to indicate error type
        for spine in ax.spines.values():
            spine.set_edgecolor("#e74c3c")
            spine.set_linewidth(3)

        text = r.get("text", "")
        if len(text) > 80:
            text = text[:77] + "..."
        ax.set_xlabel(text, fontsize=7, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_confidence_distribution(results: list[dict], save_path: str) -> None:
    """Plot probability distributions for correct vs incorrect predictions."""
    correct_probs = [r["prob"] for r in results if r["correct"]]
    wrong_probs = [r["prob"] for r in results if not r["correct"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: distribution by correctness
    axes[0].hist(correct_probs, bins=30, alpha=0.6, color="#27ae60",
                 label=f"Correct (n={len(correct_probs)})", edgecolor="black", linewidth=0.3)
    axes[0].hist(wrong_probs, bins=30, alpha=0.6, color="#e74c3c",
                 label=f"Wrong (n={len(wrong_probs)})", edgecolor="black", linewidth=0.3)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold")
    axes[0].set_xlabel("P(hateful)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Prediction Confidence: Correct vs Wrong", fontweight="bold")
    axes[0].legend()

    # Right: error types
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]  # missed hateful
    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]  # false alarm
    fn_probs = [r["prob"] for r in fn]
    fp_probs = [r["prob"] for r in fp]

    axes[1].hist(fn_probs, bins=20, alpha=0.6, color="#e67e22",
                 label=f"False Negative (n={len(fn)})", edgecolor="black", linewidth=0.3)
    axes[1].hist(fp_probs, bins=20, alpha=0.6, color="#9b59b6",
                 label=f"False Positive (n={len(fp)})", edgecolor="black", linewidth=0.3)
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold")
    axes[1].set_xlabel("P(hateful)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error Types: FN (missed hateful) vs FP (false alarm)", fontweight="bold")
    axes[1].legend()

    fig.suptitle("Error Analysis: Confidence Distribution", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Error analysis on dev set")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--top-k", type=int, default=8,
                        help="Number of worst failures to show per category")
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
    processor = CLIPProcessor.from_pretrained(clip_name)
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Load classifier
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
    logger.info("Model loaded (epoch %d)", ckpt.get("epoch", -1))

    # Load dev data
    data_dir = cfg["data"]["data_dir"]
    dev_path = Path(data_dir) / cfg["data"]["dev_file"]
    records = []
    with open(dev_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Dev set: %d samples", len(records))

    # Run inference
    results = run_inference(
        records, data_dir, model, clip_model, processor, tokenizer, device,
        batch_size=cfg["training"]["batch_size"],
    )

    # Overall metrics
    labels = [r["label"] for r in results]
    probs = [r["prob"] for r in results]
    preds = [r["pred"] for r in results]
    auroc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    correct = [r for r in results if r["correct"]]
    wrong = [r for r in results if not r["correct"]]
    fn = [r for r in results if r["label"] == 1 and r["pred"] == 0]
    fp = [r for r in results if r["label"] == 0 and r["pred"] == 1]

    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total: %d | Correct: %d (%.1f%%) | Wrong: %d (%.1f%%)",
                len(results), len(correct), len(correct)/len(results)*100,
                len(wrong), len(wrong)/len(results)*100)
    logger.info("  False Negatives (missed hateful): %d", len(fn))
    logger.info("  False Positives (false alarm):    %d", len(fp))
    logger.info("  AUROC=%.4f  Acc=%.4f  F1=%.4f", auroc, acc, f1)

    # Output directory
    out_dir = Path("outputs/error_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort failures by confidence (most confident wrong predictions first)
    fn_sorted = sorted(fn, key=lambda r: r["prob"])         # lowest prob = most confident FN
    fp_sorted = sorted(fp, key=lambda r: r["prob"], reverse=True)  # highest prob = most confident FP

    k = args.top_k

    # Plot grids
    plot_failure_grid(
        fn_sorted[:k], data_dir,
        f"False Negatives: Hateful memes predicted as NOT hateful (top {k})",
        str(out_dir / "false_negatives.png"),
    )

    plot_failure_grid(
        fp_sorted[:k], data_dir,
        f"False Positives: NOT hateful memes predicted as hateful (top {k})",
        str(out_dir / "false_positives.png"),
    )

    # Confidence distribution
    plot_confidence_distribution(results, str(out_dir / "confidence_distribution.png"))

    # Save detailed error report
    report_path = out_dir / "error_report.txt"
    with open(report_path, "w") as f:
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dev set: {len(results)} samples\n")
        f.write(f"Correct: {len(correct)} ({len(correct)/len(results)*100:.1f}%)\n")
        f.write(f"Wrong:   {len(wrong)} ({len(wrong)/len(results)*100:.1f}%)\n")
        f.write(f"  - False Negatives (missed hateful): {len(fn)}\n")
        f.write(f"  - False Positives (false alarm):    {len(fp)}\n\n")
        f.write(f"AUROC: {auroc:.4f}  Accuracy: {acc:.4f}  F1: {f1:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"TOP {k} FALSE NEGATIVES (hateful predicted as not hateful)\n")
        f.write("-" * 60 + "\n")
        for r in fn_sorted[:k]:
            f.write(f"  ID: {r['id']}  P(hateful)={r['prob']:.4f}\n")
            f.write(f"    Text: {r.get('text', '')}\n\n")

        f.write("-" * 60 + "\n")
        f.write(f"TOP {k} FALSE POSITIVES (not hateful predicted as hateful)\n")
        f.write("-" * 60 + "\n")
        for r in fp_sorted[:k]:
            f.write(f"  ID: {r['id']}  P(hateful)={r['prob']:.4f}\n")
            f.write(f"    Text: {r.get('text', '')}\n\n")

    logger.info("Error report saved to %s", report_path)
    logger.info("All outputs saved to %s/", out_dir)


if __name__ == "__main__":
    main()
