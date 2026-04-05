"""Training loop for the Hateful Meme Classifier."""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from transformers import CLIPModel, CLIPTokenizer

from src.dataset import HatefulMemesDataset
from src.model import HatefulMemeClassifier

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str) -> None:
    """Configure logging to console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logger.info("Logging to %s", log_file)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("Config loaded from %s", config_path)
    return cfg


def compute_hateful_embedding(clip_model: CLIPModel, clip_model_name: str, device: torch.device) -> torch.Tensor:
    """Compute CLIP text embedding for the word 'hateful'.

    Args:
        clip_model: Loaded CLIP model.
        clip_model_name: Model name for tokenizer.
        device: Torch device.

    Returns:
        Tensor of shape [clip_dim].
    """
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    tokens = tokenizer("hateful", return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_out = clip_model.text_model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        emb = clip_model.text_projection(text_out.pooler_output)
    return emb.squeeze(0)  # [clip_dim]


def encode_batch(
    clip_model: CLIPModel,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch through frozen CLIP encoders.

    Args:
        clip_model: Frozen CLIP model.
        pixel_values: [B, 3, 224, 224].
        input_ids: [B, seq_len].
        attention_mask: [B, seq_len].

    Returns:
        Tuple of (img_emb, txt_emb), each [B, clip_dim].
    """
    with torch.no_grad():
        vis_out = clip_model.vision_model(pixel_values=pixel_values)
        img_emb = clip_model.visual_projection(vis_out.pooler_output)
        txt_out = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_emb = clip_model.text_projection(txt_out.pooler_output)
    return img_emb, txt_emb


def run_epoch(
    model: nn.Module,
    clip_model: CLIPModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    scaler: GradScaler | None = None,
    gradient_clip: float = 1.0,
    use_amp: bool = False,
    split_name: str = "train",
) -> dict[str, float]:
    """Run a single training or evaluation epoch.

    Args:
        model: The classifier model.
        clip_model: Frozen CLIP model for encoding.
        loader: DataLoader for this split.
        criterion: Loss function.
        device: Torch device.
        optimizer: Optimizer (None for eval mode).
        scheduler: LR scheduler (None for eval mode).
        scaler: GradScaler for mixed precision.
        gradient_clip: Max gradient norm.
        use_amp: Whether to use automatic mixed precision.
        split_name: Name for logging (train / test).

    Returns:
        Dictionary with loss, auroc, accuracy, f1.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=split_name, leave=False):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()

            img_emb, txt_emb = encode_batch(clip_model, pixel_values, input_ids, attention_mask)

            if is_train and use_amp:
                with autocast(device_type=device.type):
                    logits = model(img_emb, txt_emb)
                    loss = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(img_emb, txt_emb)
                loss = criterion(logits, labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

            if is_train and scheduler is not None:
                scheduler.step()

            probs = torch.sigmoid(logits.detach()).cpu().tolist()
            preds = [1 if p >= 0.5 else 0 for p in probs]
            total_loss += loss.item() * labels.size(0)
            all_labels.extend(labels.cpu().int().tolist())
            all_probs.extend(probs)
            all_preds.extend(preds)

    n = len(all_labels)
    avg_loss = total_loss / n
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return {"loss": avg_loss, "auroc": auroc, "accuracy": acc, "f1": f1}


def train(config_path: str = "config/config.yaml") -> None:
    """Main training entry point."""
    cfg = load_config(config_path)
    setup_logging("logs")

    seed = cfg["training"]["seed"]
    set_seed(seed)
    logger.info("Seed set to %d", seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    use_amp = device.type == "cuda"
    num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    # Data
    data_dir = cfg["data"]["data_dir"]
    clip_model_name = cfg["model"]["clip_model"]

    train_ds = HatefulMemesDataset(
        jsonl_path=str(Path(data_dir) / cfg["data"]["train_file"]),
        data_dir=data_dir,
        clip_model_name=clip_model_name,
    )
    dev_ds = HatefulMemesDataset(
        jsonl_path=str(Path(data_dir) / cfg["data"]["dev_file"]),
        data_dir=data_dir,
        clip_model_name=clip_model_name,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Frozen CLIP encoder
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False
    logger.info("CLIP encoder loaded and frozen")

    # Compute hateful embedding for query initialization
    hateful_emb = compute_hateful_embedding(clip_model, clip_model_name, device)
    logger.info("Hateful query embedding computed, shape: %s", hateful_emb.shape)

    # Class weights → pos_weight for BCEWithLogitsLoss
    class_weights = train_ds.get_class_weights()
    pos_weight = (class_weights[1] / class_weights[0]).to(device)
    logger.info("pos_weight: %.4f", pos_weight.item())

    # Model
    model = HatefulMemeClassifier(
        clip_dim=cfg["model"]["clip_dim"],
        hidden=cfg["model"]["hidden"],
        num_heads=cfg["model"]["num_heads"],
        hateful_emb=hateful_emb,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    # Scheduler: linear warmup then cosine decay
    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = cfg["training"]["warmup_steps"]
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    scaler = GradScaler() if use_amp else None

    # Training loop
    best_auroc = 0.0
    patience_counter = 0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        logger.info("===== Epoch %d / %d =====", epoch, cfg["training"]["epochs"])

        train_metrics = run_epoch(
            model, clip_model, train_loader, criterion, device,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            gradient_clip=cfg["training"]["gradient_clip"],
            use_amp=use_amp, split_name="train",
        )
        logger.info(
            "Train — loss: %.4f | AUROC: %.4f | acc: %.4f | F1: %.4f",
            train_metrics["loss"], train_metrics["auroc"],
            train_metrics["accuracy"], train_metrics["f1"],
        )

        dev_metrics = run_epoch(
            model, clip_model, dev_loader, criterion, device,
            split_name="dev", use_amp=use_amp,
        )
        logger.info(
            "Dev    — loss: %.4f | AUROC: %.4f | acc: %.4f | F1: %.4f",
            dev_metrics["loss"], dev_metrics["auroc"],
            dev_metrics["accuracy"], dev_metrics["f1"],
        )

        # Checkpoint & early stopping
        if dev_metrics["auroc"] > best_auroc:
            best_auroc = dev_metrics["auroc"]
            patience_counter = 0
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_auroc": best_auroc,
                "config": cfg,
            }, ckpt_path)
            logger.info("New best model saved (AUROC=%.4f) → %s", best_auroc, ckpt_path)
        else:
            patience_counter += 1
            logger.info(
                "No improvement (%d / %d patience)",
                patience_counter, cfg["training"]["patience"],
            )
            if patience_counter >= cfg["training"]["patience"]:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    logger.info("Training complete. Best dev AUROC: %.4f", best_auroc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hateful Meme Classifier")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()
    train(args.config)
