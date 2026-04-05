"""Exploratory Data Analysis for the Hateful Memes dataset.

Generates a comprehensive set of visualisations and statistics saved to
`outputs/eda/` (created automatically). Designed to run standalone:

    python scripts/eda.py

All plots are saved as PNG; a summary report is printed to stdout and
saved as `outputs/eda/eda_report.txt`.
"""

import json
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works headless / SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_DIR = Path("outputs/eda")
SPLITS = {
    "train": DATA_DIR / "train.jsonl",
    "dev": DATA_DIR / "dev.jsonl",
    "test": DATA_DIR / "test.jsonl",
}
IMG_DIR = DATA_DIR / "img"

# Palette
C_HATEFUL = "#e74c3c"
C_NOT = "#2ecc71"
C_UNKNOWN = "#95a5a6"
PALETTE = [C_NOT, C_HATEFUL]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_fig(fig, name: str) -> None:
    """Save a matplotlib figure to OUT_DIR/<name>.png."""
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ saved {path}")


report_lines: list[str] = []

def report(msg: str = "") -> None:
    """Print and buffer a report line."""
    print(msg)
    report_lines.append(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dataset overview
# ═══════════════════════════════════════════════════════════════════════════════

def dataset_overview(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("DATASET OVERVIEW")
    report("=" * 70)
    for name, records in splits.items():
        n = len(records)
        has_label = sum(1 for r in records if "label" in r)
        has_text = sum(1 for r in records if r.get("text"))
        report(f"  {name:6s}: {n:>6,} samples | {has_label:>6,} labelled | {has_text:>6,} with text")
    total = sum(len(r) for r in splits.values())
    report(f"  {'TOTAL':6s}: {total:>6,}")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Label distribution
# ═══════════════════════════════════════════════════════════════════════════════

def label_distribution(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("LABEL DISTRIBUTION")
    report("=" * 70)

    labelled_splits = {k: v for k, v in splits.items()
                       if any("label" in r for r in v)}

    fig, axes = plt.subplots(1, len(labelled_splits), figsize=(5 * len(labelled_splits), 5))
    if len(labelled_splits) == 1:
        axes = [axes]

    for ax, (name, records) in zip(axes, labelled_splits.items()):
        labels = [r["label"] for r in records if "label" in r]
        counts = Counter(labels)
        not_h = counts.get(0, 0)
        hat_h = counts.get(1, 0)
        total = not_h + hat_h
        ratio = hat_h / total * 100 if total else 0

        bars = ax.bar(["Not Hateful (0)", "Hateful (1)"], [not_h, hat_h],
                       color=PALETTE, edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, [not_h, hat_h]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                    f"{val}\n({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{name} split (n={total:,})", fontsize=13, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(not_h, hat_h) * 1.25)

        report(f"  {name}: not_hateful={not_h}  hateful={hat_h}  ratio={ratio:.1f}%  imbalance={not_h/(hat_h or 1):.2f}:1")

    fig.suptitle("Label Distribution by Split", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, "01_label_distribution")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Text length analysis
# ═══════════════════════════════════════════════════════════════════════════════

def text_length_analysis(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("TEXT LENGTH ANALYSIS")
    report("=" * 70)

    # --- Per-split histograms ---
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5), sharey=True)
    if len(splits) == 1:
        axes = [axes]

    for ax, (name, records) in zip(axes, splits.items()):
        texts = [r.get("text", "") for r in records]
        word_counts = [len(t.split()) for t in texts]
        char_counts = [len(t) for t in texts]

        ax.hist(word_counts, bins=40, color="#3498db", edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_title(f"{name} (n={len(records):,})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Word count")
        ax.set_ylabel("Frequency")

        mean_w = np.mean(word_counts) if word_counts else 0
        med_w = np.median(word_counts) if word_counts else 0
        mean_c = np.mean(char_counts) if char_counts else 0
        ax.axvline(mean_w, color="red", linestyle="--", linewidth=1.2, label=f"mean={mean_w:.1f}")
        ax.axvline(med_w, color="orange", linestyle="--", linewidth=1.2, label=f"median={med_w:.0f}")
        ax.legend(fontsize=9)

        report(f"  {name}: words  mean={mean_w:.1f}  median={med_w:.0f}  "
               f"min={min(word_counts)}  max={max(word_counts)}  |  "
               f"chars  mean={mean_c:.1f}")

    fig.suptitle("Text Length Distribution (words)", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, "02_text_length_words")

    # --- Hateful vs Not by word length (train only) ---
    train = splits.get("train", [])
    if train and "label" in train[0]:
        fig, ax = plt.subplots(figsize=(8, 5))
        wc_0 = [len(r["text"].split()) for r in train if r.get("label") == 0 and r.get("text")]
        wc_1 = [len(r["text"].split()) for r in train if r.get("label") == 1 and r.get("text")]
        bins = np.arange(0, max(max(wc_0, default=0), max(wc_1, default=0)) + 2)
        ax.hist(wc_0, bins=bins, alpha=0.6, color=C_NOT, label=f"Not Hateful (n={len(wc_0)})", edgecolor="black", linewidth=0.3)
        ax.hist(wc_1, bins=bins, alpha=0.6, color=C_HATEFUL, label=f"Hateful (n={len(wc_1)})", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("Word count")
        ax.set_ylabel("Frequency")
        ax.set_title("Train: Text Length by Label", fontsize=13, fontweight="bold")
        ax.legend()
        save_fig(fig, "03_text_length_by_label")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Image dimension analysis
# ═══════════════════════════════════════════════════════════════════════════════

def image_dimensions(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("IMAGE DIMENSIONS")
    report("=" * 70)

    all_records = []
    for records in splits.values():
        all_records.extend(records)

    widths, heights, aspects = [], [], []
    sample_n = min(len(all_records), 3000)  # sample for speed
    rng = np.random.RandomState(42)
    sampled = rng.choice(len(all_records), sample_n, replace=False)

    for idx in sampled:
        r = all_records[idx]
        img_path = DATA_DIR / r["img"]
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            widths.append(w)
            heights.append(h)
            aspects.append(w / h)
        except Exception:
            pass

    if not widths:
        report("  ⚠ No images found, skipping dimension analysis")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(widths, bins=40, color="#9b59b6", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Width", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Pixels")

    axes[1].hist(heights, bins=40, color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Height", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Pixels")

    axes[2].hist(aspects, bins=40, color="#1abc9c", edgecolor="black", linewidth=0.5)
    axes[2].set_title("Aspect Ratio (W/H)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Ratio")

    fig.suptitle(f"Image Dimensions (sampled {sample_n:,} images)", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, "04_image_dimensions")

    report(f"  Width:   mean={np.mean(widths):.0f}  median={np.median(widths):.0f}  "
           f"min={min(widths)}  max={max(widths)}")
    report(f"  Height:  mean={np.mean(heights):.0f}  median={np.median(heights):.0f}  "
           f"min={min(heights)}  max={max(heights)}")
    report(f"  Aspect:  mean={np.mean(aspects):.2f}  median={np.median(aspects):.2f}")

    # Scatter: width vs height
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(widths, heights, alpha=0.25, s=8, color="#3498db")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Width vs Height", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    save_fig(fig, "05_width_vs_height")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Top words / vocabulary analysis
# ═══════════════════════════════════════════════════════════════════════════════

def vocabulary_analysis(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("VOCABULARY ANALYSIS (train)")
    report("=" * 70)

    train = splits.get("train", [])
    if not train:
        report("  ⚠ No train split found")
        return

    import re
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "could", "should", "may", "might", "shall", "can",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from",
                 "as", "into", "through", "during", "before", "after",
                 "and", "but", "or", "nor", "not", "no", "so", "if", "then",
                 "than", "too", "very", "just", "about", "above", "below",
                 "it", "its", "this", "that", "these", "those",
                 "i", "me", "my", "we", "us", "our", "you", "your",
                 "he", "him", "his", "she", "her", "they", "them", "their",
                 "who", "whom", "which", "what", "when", "where", "how",
                 "all", "each", "every", "both", "few", "more", "most",
                 "other", "some", "such", "only", "own", "same",
                 "don't", "doesn't", "didn't", "won't", "wouldn't",
                 "can't", "couldn't", "shouldn't", "isn't", "aren't",
                 "wasn't", "weren't", "hasn't", "haven't", "hadn't",
                 "don", "doesn", "didn", "won", "wouldn", "couldn",
                 "get", "got", "like", "one", "also", "even", "still",
                 "up", "out", "over", "down", "off", "here", "there"}

    def tokenize(text: str) -> list[str]:
        return [w.lower() for w in re.findall(r"[a-zA-Z']+", text) if len(w) > 1]

    # Overall word freq
    all_words = []
    for r in train:
        all_words.extend(tokenize(r.get("text", "")))
    total_vocab = len(set(all_words))
    report(f"  Total tokens: {len(all_words):,}   Unique: {total_vocab:,}")

    # Top words (no stopwords)
    content_words = [w for w in all_words if w not in stopwords]
    top30 = Counter(content_words).most_common(30)

    fig, ax = plt.subplots(figsize=(10, 7))
    words_list = [w for w, c in top30][::-1]
    counts_list = [c for w, c in top30][::-1]
    ax.barh(words_list, counts_list, color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Frequency")
    ax.set_title("Top 30 Content Words (train, stopwords removed)", fontsize=13, fontweight="bold")
    save_fig(fig, "06_top_words_overall")

    report(f"  Top 10 words: {', '.join(f'{w}({c})' for w, c in top30[:10])}")

    # Per-label word frequency
    if "label" in train[0]:
        words_0 = []
        words_1 = []
        for r in train:
            toks = tokenize(r.get("text", ""))
            toks = [w for w in toks if w not in stopwords]
            if r.get("label") == 0:
                words_0.extend(toks)
            else:
                words_1.extend(toks)

        freq_0 = Counter(words_0)
        freq_1 = Counter(words_1)

        # Words most distinctive to hateful class (log-odds ratio)
        all_content = set(freq_0.keys()) | set(freq_1.keys())
        n0, n1 = max(len(words_0), 1), max(len(words_1), 1)
        scores = {}
        for w in all_content:
            if freq_0.get(w, 0) + freq_1.get(w, 0) < 5:
                continue  # skip rare words
            rate_0 = (freq_0.get(w, 0) + 1) / (n0 + 1)
            rate_1 = (freq_1.get(w, 0) + 1) / (n1 + 1)
            scores[w] = np.log2(rate_1 / rate_0)

        top_hateful = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:25]
        top_not = sorted(scores.items(), key=lambda x: x[1])[:25]

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Most hateful-leaning words
        hw = [w for w, s in top_hateful][::-1]
        hs = [s for w, s in top_hateful][::-1]
        axes[0].barh(hw, hs, color=C_HATEFUL, edgecolor="black", linewidth=0.5)
        axes[0].set_xlabel("Log₂ Odds Ratio (hateful / not)")
        axes[0].set_title("Words most associated with HATEFUL", fontsize=12, fontweight="bold")

        # Most not-hateful-leaning words
        nw = [w for w, s in top_not][::-1]
        ns = [abs(s) for w, s in top_not][::-1]
        axes[1].barh(nw, ns, color=C_NOT, edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Log₂ Odds Ratio (not / hateful)")
        axes[1].set_title("Words most associated with NOT HATEFUL", fontsize=12, fontweight="bold")

        fig.suptitle("Distinctive Words by Label (Log-Odds Ratio)", fontsize=14, fontweight="bold", y=1.02)
        save_fig(fig, "07_distinctive_words_by_label")

        report(f"  Top hateful-leaning: {', '.join(w for w, _ in top_hateful[:10])}")
        report(f"  Top non-hateful-leaning: {', '.join(w for w, _ in top_not[:10])}")

    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Text overlap between splits
# ═══════════════════════════════════════════════════════════════════════════════

def text_overlap(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("TEXT OVERLAP BETWEEN SPLITS")
    report("=" * 70)

    texts_by_split = {}
    for name, records in splits.items():
        texts_by_split[name] = set(r.get("text", "").strip().lower() for r in records)

    names = list(texts_by_split.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = texts_by_split[a] & texts_by_split[b]
            report(f"  {a} ∩ {b}: {len(overlap)} overlapping texts "
                   f"({len(overlap)/min(len(texts_by_split[a]), len(texts_by_split[b]))*100:.1f}% of smaller)")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Sample memes grid
# ═══════════════════════════════════════════════════════════════════════════════

def sample_memes_grid(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("SAMPLE MEMES")
    report("=" * 70)

    train = splits.get("train", [])
    if not train or "label" not in train[0]:
        return

    hateful = [r for r in train if r.get("label") == 1]
    not_hateful = [r for r in train if r.get("label") == 0]

    rng = np.random.RandomState(42)
    n_per_class = 6

    samples_h = [hateful[i] for i in rng.choice(len(hateful), min(n_per_class, len(hateful)), replace=False)]
    samples_n = [not_hateful[i] for i in rng.choice(len(not_hateful), min(n_per_class, len(not_hateful)), replace=False)]

    fig, axes = plt.subplots(2, n_per_class, figsize=(4 * n_per_class, 10))

    for col, r in enumerate(samples_n):
        ax = axes[0, col]
        img_path = DATA_DIR / r["img"]
        if img_path.exists():
            im = Image.open(img_path)
            ax.imshow(im)
        ax.set_title(f"NOT HATEFUL\nid={r['id']}", fontsize=9, color=C_NOT, fontweight="bold")
        txt = r.get("text", "")[:60]
        ax.set_xlabel(textwrap.fill(txt, 25), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for col, r in enumerate(samples_h):
        ax = axes[1, col]
        img_path = DATA_DIR / r["img"]
        if img_path.exists():
            im = Image.open(img_path)
            ax.imshow(im)
        ax.set_title(f"HATEFUL\nid={r['id']}", fontsize=9, color=C_HATEFUL, fontweight="bold")
        txt = r.get("text", "")[:60]
        ax.set_xlabel(textwrap.fill(txt, 25), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Random Sample Memes", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "08_sample_memes")
    report(f"  Saved 2×{n_per_class} sample meme grid")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Character-level analysis & special patterns
# ═══════════════════════════════════════════════════════════════════════════════

def text_patterns(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("TEXT PATTERNS (train)")
    report("=" * 70)

    train = splits.get("train", [])
    if not train:
        return

    texts = [r.get("text", "") for r in train]

    # Empty / very short texts
    empty = sum(1 for t in texts if len(t.strip()) == 0)
    short = sum(1 for t in texts if 0 < len(t.split()) <= 3)
    long_ = sum(1 for t in texts if len(t.split()) > 30)

    # ALL CAPS texts
    allcaps = sum(1 for t in texts if t == t.upper() and len(t) > 5)

    # Texts with URLs, hashtags, @mentions
    import re
    has_url = sum(1 for t in texts if re.search(r"https?://|www\.", t))
    has_hash = sum(1 for t in texts if "#" in t)
    has_mention = sum(1 for t in texts if "@" in t)

    # Questions / exclamations
    has_question = sum(1 for t in texts if "?" in t)
    has_exclaim = sum(1 for t in texts if "!" in t)

    n = len(texts)
    report(f"  Empty texts:     {empty:>5} ({empty/n*100:.1f}%)")
    report(f"  Short (≤3 words):{short:>5} ({short/n*100:.1f}%)")
    report(f"  Long (>30 words):{long_:>5} ({long_/n*100:.1f}%)")
    report(f"  ALL CAPS:        {allcaps:>5} ({allcaps/n*100:.1f}%)")
    report(f"  Contains URL:    {has_url:>5} ({has_url/n*100:.1f}%)")
    report(f"  Contains #:      {has_hash:>5} ({has_hash/n*100:.1f}%)")
    report(f"  Contains @:      {has_mention:>5} ({has_mention/n*100:.1f}%)")
    report(f"  Questions (?):   {has_question:>5} ({has_question/n*100:.1f}%)")
    report(f"  Exclamations (!): {has_exclaim:>4} ({has_exclaim/n*100:.1f}%)")

    # Bar chart of patterns
    labels_p = ["Empty", "Short\n(≤3w)", "Long\n(>30w)", "ALL\nCAPS",
                "URL", "#", "@", "?", "!"]
    vals = [empty, short, long_, allcaps, has_url, has_hash, has_mention,
            has_question, has_exclaim]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels_p, vals, color="#3498db", edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + n * 0.005,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Text Pattern Counts (train)", fontsize=13, fontweight="bold")
    save_fig(fig, "09_text_patterns")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Bigram analysis
# ═══════════════════════════════════════════════════════════════════════════════

def bigram_analysis(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("TOP BIGRAMS (train)")
    report("=" * 70)

    train = splits.get("train", [])
    if not train:
        return

    import re
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
                 "in", "for", "on", "with", "at", "by", "from", "and", "but", "or",
                 "it", "its", "this", "that", "i", "me", "my", "you", "your", "he",
                 "she", "they", "them", "we", "us", "our", "his", "her", "their"}

    bigrams_all = Counter()
    bigrams_0 = Counter()
    bigrams_1 = Counter()

    for r in train:
        words = [w.lower() for w in re.findall(r"[a-zA-Z']+", r.get("text", "")) if len(w) > 1]
        words = [w for w in words if w not in stopwords]
        bgs = list(zip(words[:-1], words[1:]))
        bigrams_all.update(bgs)
        if r.get("label") == 0:
            bigrams_0.update(bgs)
        elif r.get("label") == 1:
            bigrams_1.update(bgs)

    top20 = bigrams_all.most_common(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    bg_labels = [f"{a} {b}" for (a, b), c in top20][::-1]
    bg_counts = [c for (a, b), c in top20][::-1]
    ax.barh(bg_labels, bg_counts, color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Frequency")
    ax.set_title("Top 20 Bigrams (train, stopwords removed)", fontsize=13, fontweight="bold")
    save_fig(fig, "10_top_bigrams")

    report(f"  Top 10: {', '.join(f'{a} {b}({c})' for (a, b), c in top20[:10])}")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Correlation: text length vs label
# ═══════════════════════════════════════════════════════════════════════════════

def length_vs_label(splits: dict[str, list[dict]]) -> None:
    report("=" * 70)
    report("TEXT LENGTH vs LABEL (train)")
    report("=" * 70)

    train = splits.get("train", [])
    if not train or "label" not in train[0]:
        return

    wc_0 = [len(r["text"].split()) for r in train if r.get("label") == 0 and r.get("text")]
    wc_1 = [len(r["text"].split()) for r in train if r.get("label") == 1 and r.get("text")]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([wc_0, wc_1], tick_labels=["Not Hateful (0)", "Hateful (1)"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C_NOT)
    bp["boxes"][1].set_facecolor(C_HATEFUL)
    for box in bp["boxes"]:
        box.set_edgecolor("black")
    ax.set_ylabel("Word Count")
    ax.set_title("Text Length Distribution by Label", fontsize=13, fontweight="bold")
    save_fig(fig, "11_length_vs_label_boxplot")

    report(f"  Not Hateful: mean={np.mean(wc_0):.1f}  median={np.median(wc_0):.0f}  std={np.std(wc_0):.1f}")
    report(f"  Hateful:     mean={np.mean(wc_1):.1f}  median={np.median(wc_1):.0f}  std={np.std(wc_1):.1f}")
    report()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report("Hateful Memes — Exploratory Data Analysis")
    report(f"Output directory: {OUT_DIR.resolve()}")
    report()

    # Load data
    splits: dict[str, list[dict]] = {}
    for name, path in SPLITS.items():
        if path.exists():
            splits[name] = load_jsonl(path)
            print(f"  Loaded {name}: {len(splits[name]):,} records")
        else:
            print(f"  ⚠ {path} not found, skipping")

    if not splits:
        print("ERROR: No data found. Make sure data/ directory has train.jsonl, dev.jsonl, test.jsonl")
        sys.exit(1)

    print()

    # Run all analyses
    dataset_overview(splits)
    label_distribution(splits)
    text_length_analysis(splits)
    image_dimensions(splits)
    vocabulary_analysis(splits)
    text_overlap(splits)
    sample_memes_grid(splits)
    text_patterns(splits)
    bigram_analysis(splits)
    length_vs_label(splits)

    # Save report
    report_path = OUT_DIR / "eda_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Full report saved to {report_path}")
    print(f"✓ All plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
