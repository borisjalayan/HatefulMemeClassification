"""Dataset class for Facebook Hateful Memes challenge."""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPTokenizer

logger = logging.getLogger(__name__)


class HatefulMemesDataset(Dataset):
    """PyTorch Dataset for the Hateful Memes benchmark.

    Loads JSONL annotations, applies CLIP preprocessing to images,
    and tokenizes meme text with CLIPTokenizer.
    """

    def __init__(
        self,
        jsonl_path: str,
        data_dir: str,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
    ) -> None:
        """Initialise the dataset.

        Args:
            jsonl_path: Path to the JSONL annotation file.
            data_dir: Root data directory (image paths are resolved relative to this).
            clip_model_name: HuggingFace CLIP model identifier.
            max_length: Maximum token length for text inputs.
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        self.samples: list[dict[str, Any]] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.samples.append(entry)

        logger.info("Loaded %d samples from %s", len(self.samples), jsonl_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        img_path = self.data_dir / sample["img"]
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        text_inputs = self.tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        label = sample.get("label", -1)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "id": sample["id"],
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced training.

        Returns:
            Tensor of shape [num_classes] with weight per class.
        """
        labels = [s["label"] for s in self.samples if "label" in s]
        counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * len(weights)
        logger.info("Class counts: %s  →  weights: %s", counts.tolist(), weights.tolist())
        return weights
