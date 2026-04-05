"""Parse training logs and generate loss / metric curves.

Scans all training log files in logs/ for epoch-level metrics and
produces publication-ready plots saved to outputs/.

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --log logs/train_20260405_132034.log
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path: str) -> dict:
    """Parse a training log file and extract per-epoch metrics.

    Returns dict with keys: train_loss, train_auroc, train_acc, train_f1,
                             dev_loss, dev_auroc, dev_acc, dev_f1, epochs
    """
    # Patterns:
    # Train — loss: 0.7602 | AUROC: 0.7497 | acc: 0.6762 | F1: 0.6657
    # Dev    — loss: 0.7993 | AUROC: 0.7282 | acc: 0.6640 | F1: 0.6569
    train_pattern = re.compile(
        r"Train\s*—\s*loss:\s*([\d.]+)\s*\|\s*AUROC:\s*([\d.]+)\s*\|\s*acc:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)"
    )
    dev_pattern = re.compile(
        r"(?:Dev|Test)\s*—\s*loss:\s*([\d.]+)\s*\|\s*AUROC:\s*([\d.]+)\s*\|\s*acc:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)"
    )

    data = {
        "train_loss": [], "train_auroc": [], "train_acc": [], "train_f1": [],
        "dev_loss": [], "dev_auroc": [], "dev_acc": [], "dev_f1": [],
    }

    with open(log_path) as f:
        for line in f:
            m = train_pattern.search(line)
            if m:
                data["train_loss"].append(float(m.group(1)))
                data["train_auroc"].append(float(m.group(2)))
                data["train_acc"].append(float(m.group(3)))
                data["train_f1"].append(float(m.group(4)))
                continue

            m = dev_pattern.search(line)
            if m:
                data["dev_loss"].append(float(m.group(1)))
                data["dev_auroc"].append(float(m.group(2)))
                data["dev_acc"].append(float(m.group(3)))
                data["dev_f1"].append(float(m.group(4)))

    n_epochs = len(data["train_loss"])
    data["epochs"] = list(range(1, n_epochs + 1))

    # Trim dev to match train length (in case of incomplete final epoch)
    for key in ["dev_loss", "dev_auroc", "dev_acc", "dev_f1"]:
        data[key] = data[key][:n_epochs]

    return data


def plot_curves(data: dict, out_dir: str = "outputs") -> None:
    """Generate and save training curve plots."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = data["epochs"]

    if not epochs:
        print("ERROR: No epoch data found in log file")
        sys.exit(1)

    # ── 1. Loss curves ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, data["train_loss"], "o-", color="#3498db", linewidth=2,
            markersize=6, label="Train Loss")
    if data["dev_loss"]:
        ax.plot(epochs, data["dev_loss"], "s-", color="#e74c3c", linewidth=2,
                markersize=6, label="Dev Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCEWithLogits)", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.savefig(out / "training_loss_curve.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {out / 'training_loss_curve.png'}")

    # ── 2. AUROC curves ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, data["train_auroc"], "o-", color="#3498db", linewidth=2,
            markersize=6, label="Train AUROC")
    if data["dev_auroc"]:
        ax.plot(epochs, data["dev_auroc"], "s-", color="#e74c3c", linewidth=2,
                markersize=6, label="Dev AUROC")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("AUROC Over Training", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.set_ylim(0.5, 1.0)
    fig.savefig(out / "training_auroc_curve.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {out / 'training_auroc_curve.png'}")

    # ── 3. Accuracy & F1 curves ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Accuracy
    axes[0].plot(epochs, data["train_acc"], "o-", color="#3498db", linewidth=2,
                 markersize=6, label="Train Acc")
    if data["dev_acc"]:
        axes[0].plot(epochs, data["dev_acc"], "s-", color="#e74c3c", linewidth=2,
                     markersize=6, label="Dev Acc")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)

    # F1
    axes[1].plot(epochs, data["train_f1"], "o-", color="#3498db", linewidth=2,
                 markersize=6, label="Train F1")
    if data["dev_f1"]:
        axes[1].plot(epochs, data["dev_f1"], "s-", color="#e74c3c", linewidth=2,
                     markersize=6, label="Dev F1")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Macro F1", fontsize=12)
    axes[1].set_title("Macro F1 Score", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)

    fig.suptitle("Accuracy & F1 Over Training", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(out / "training_acc_f1_curves.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {out / 'training_acc_f1_curves.png'}")

    # ── 4. All-in-one summary ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("Loss", "train_loss", "dev_loss", False),
        ("AUROC", "train_auroc", "dev_auroc", True),
        ("Accuracy", "train_acc", "dev_acc", True),
        ("Macro F1", "train_f1", "dev_f1", True),
    ]

    for ax, (name, train_key, dev_key, set_ylim) in zip(axes.flat, metrics):
        ax.plot(epochs, data[train_key], "o-", color="#3498db", linewidth=2,
                markersize=5, label="Train")
        if data[dev_key]:
            ax.plot(epochs, data[dev_key], "s-", color="#e74c3c", linewidth=2,
                    markersize=5, label="Dev")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        if set_ylim:
            ax.set_ylim(0.4, 1.0)

    fig.suptitle("Training Summary — All Metrics", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "training_summary.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {out / 'training_summary.png'}")

    # ── Print summary table ──────────────────────────────────────────────
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Dev Loss':>10} | "
          f"{'Train AUROC':>11} | {'Dev AUROC':>10} | {'Train F1':>8} | {'Dev F1':>7}")
    print("-" * 80)
    for i, ep in enumerate(epochs):
        tl = data["train_loss"][i]
        dl = data["dev_loss"][i] if i < len(data["dev_loss"]) else float("nan")
        ta = data["train_auroc"][i]
        da = data["dev_auroc"][i] if i < len(data["dev_auroc"]) else float("nan")
        tf = data["train_f1"][i]
        df = data["dev_f1"][i] if i < len(data["dev_f1"]) else float("nan")
        print(f"{ep:>5} | {tl:>10.4f} | {dl:>10.4f} | {ta:>11.4f} | {da:>10.4f} | {tf:>8.4f} | {df:>7.4f}")

    if data["dev_auroc"]:
        best_ep = int(np.argmax(data["dev_auroc"])) + 1
        best_auroc = max(data["dev_auroc"])
        print(f"\n★ Best Dev AUROC: {best_auroc:.4f} at epoch {best_ep}")


def find_latest_log() -> str:
    """Find the most recent training log file."""
    logs = sorted(Path("logs").glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("ERROR: No training logs found in logs/")
        sys.exit(1)
    return str(logs[-1])


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from logs")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to training log file (default: latest in logs/)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="Output directory for plots")
    args = parser.parse_args()

    log_path = args.log or find_latest_log()
    print(f"Parsing: {log_path}")

    data = parse_log(log_path)
    print(f"Found {len(data['epochs'])} epochs\n")

    plot_curves(data, args.output)


if __name__ == "__main__":
    main()
