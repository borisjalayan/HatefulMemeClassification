"""Hateful meme classifier with 3-branch CLIP + cross-attention fusion."""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

logger = logging.getLogger(__name__)


class HatefulMemeClassifier(nn.Module):
    """
    Cross-attention classifier with Q=CLIP("hateful").

    The word "hateful" queries the meme's image and text embeddings
    via cross-attention to determine how hateful the content is.

    Args:
        clip_dim:     CLIP embedding dimension (768 for ViT-L/14)
        hidden:       hidden dimension for attention and MLP (256)
        num_heads:    number of attention heads (4)
        hateful_emb:  CLIP embedding of "hateful" (768d), used to initialize Q
    """

    def __init__(self, clip_dim=768, hidden=256, num_heads=4, hateful_emb=None):
        super().__init__()
        self.hidden = hidden

        # Separate K and V projections for image and text
        self.img_proj_k = nn.Linear(clip_dim, hidden)
        self.txt_proj_k = nn.Linear(clip_dim, hidden)
        self.img_proj_v = nn.Linear(clip_dim, hidden)
        self.txt_proj_v = nn.Linear(clip_dim, hidden)

        # Query: learnable, initialized from CLIP("hateful")
        self.query_proj = nn.Linear(clip_dim, hidden)
        if hateful_emb is not None:
            self.hateful_emb = nn.Parameter(hateful_emb.clone())
        else:
            self.hateful_emb = nn.Parameter(torch.randn(clip_dim))

        # Multi-head cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads,
            dropout=0.0, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden)

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(128, 1),
        )

    def forward(self, img_emb, txt_emb):
        """
        Args:
            img_emb: (B, 768) CLIP image embeddings
            txt_emb: (B, 768) CLIP text embeddings
        Returns:
            logits: (B,)
        """
        B = img_emb.size(0)

        # K and V: 2 tokens (image, text)
        K = torch.stack([self.img_proj_k(img_emb), self.txt_proj_k(txt_emb)], dim=1)  # (B, 2, hidden)
        V = torch.stack([self.img_proj_v(img_emb), self.txt_proj_v(txt_emb)], dim=1)  # (B, 2, hidden)

        # Q from "hateful" embedding
        q = self.query_proj(self.hateful_emb)                    # (hidden,)
        Q = q.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)        # (B, 1, hidden)

        # Cross-attention
        attn_out, _ = self.attn(Q, K, V)                        # (B, 1, hidden)
        attn_out = self.attn_norm(attn_out.squeeze(1))           # (B, hidden)

        return self.head(attn_out).squeeze(-1)                   # (B,)
