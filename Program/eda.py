"""
═══════════════════════════════════════════════════════════════════════
  eda.py — Exploratory Data Analysis
═══════════════════════════════════════════════════════════════════════

Reads the dataset/ folder and produces four PNG charts for the report:
  1. eda_class_distribution.png  — number of images per class (bar chart)
  2. eda_sample_grid.png         — sample face from each class (image grid)
  3. eda_image_dimensions.png    — width/height scatter (image dimension distribution)
  4. eda_pixel_intensities.png   — per-class average pixel intensity histograms

Run from inside the Site2_Attendance folder:
    python eda.py

Outputs are saved to the same folder. Drop them into the report's figure spots.
═══════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

BASE        = Path(__file__).parent
DATASET     = BASE / "dataset"
ALLOWED     = {".jpg", ".jpeg", ".png", ".webp"}
PALETTE     = ["#6ee7b7", "#818cf8", "#fbbf24", "#f87171", "#34d399",
               "#60a5fa", "#a78bfa", "#fb7185", "#fde68a", "#86efac"]


def collect_dataset():
    """Walk dataset/ and return {class_name: [paths]} mapping."""
    if not DATASET.exists():
        raise SystemExit("[ERROR] dataset/ folder not found. Run from project root.")

    data = {}
    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir():
            continue
        paths = sorted([p for p in person_dir.iterdir()
                        if p.suffix.lower() in ALLOWED])
        if paths:
            data[person_dir.name] = paths
    if not data:
        raise SystemExit("[ERROR] dataset/ is empty.")
    return data


# ════════════════════════════════════════════════════════════════════
#  CHART 1 — Class distribution bar chart
# ════════════════════════════════════════════════════════════════════
def chart_class_distribution(data):
    print("[1/4] Generating class distribution chart...")
    classes = list(data.keys())
    counts = [len(v) for v in data.values()]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    bars = ax.bar(classes, counts,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(classes))],
                  edgecolor="#1f2937", linewidth=1.5)

    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(count), ha="center", fontweight="bold", fontsize=11)

    ax.set_title("Sample Distribution per Class", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of images", fontsize=12, labelpad=10)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(BASE / "eda_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    saved -> eda_class_distribution.png")


# ════════════════════════════════════════════════════════════════════
#  CHART 2 — Sample images grid (one face per class)
# ════════════════════════════════════════════════════════════════════
def chart_sample_grid(data):
    print("[2/4] Generating sample images grid...")
    classes = list(data.keys())
    n = len(classes)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2), dpi=120)
    axes = np.array(axes).reshape(rows, cols) if rows > 1 else np.array([axes]).reshape(1, cols)

    for i, cls in enumerate(classes):
        r, c = i // cols, i % cols
        # Pick the middle image as representative
        sample_path = data[cls][len(data[cls]) // 2]
        img = cv2.imread(str(sample_path))
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ax = axes[r][c]
        ax.imshow(img_rgb)
        ax.set_title(f"{cls}\n({len(data[cls])} images)",
                     fontsize=11, fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.suptitle("Sample Images per Class", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(BASE / "eda_sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    saved -> eda_sample_grid.png")


# ════════════════════════════════════════════════════════════════════
#  CHART 3 — Image dimensions scatter
# ════════════════════════════════════════════════════════════════════
def chart_image_dimensions(data):
    print("[3/4] Generating image dimensions chart...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    for i, (cls, paths) in enumerate(data.items()):
        widths, heights = [], []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None: continue
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
        ax.scatter(widths, heights,
                   color=PALETTE[i % len(PALETTE)],
                   alpha=0.6, s=50, label=cls, edgecolors="#1f2937", linewidth=0.5)

    ax.set_title("Image Dimension Distribution", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Width (pixels)", fontsize=12, labelpad=10)
    ax.set_ylabel("Height (pixels)", fontsize=12, labelpad=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(BASE / "eda_image_dimensions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    saved -> eda_image_dimensions.png")


# ════════════════════════════════════════════════════════════════════
#  CHART 4 — Pixel intensity histograms per class
# ════════════════════════════════════════════════════════════════════
def chart_pixel_intensities(data):
    print("[4/4] Generating pixel intensity histograms...")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)

    for i, (cls, paths) in enumerate(data.items()):
        # Sample up to 30 images per class (for speed)
        sample = paths[:30]
        all_pixels = []
        for p in sample:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            all_pixels.append(img.flatten())
        if not all_pixels: continue
        all_pixels = np.concatenate(all_pixels)
        ax.hist(all_pixels, bins=50, alpha=0.45,
                color=PALETTE[i % len(PALETTE)],
                label=cls, edgecolor="#1f2937", linewidth=0.5,
                density=True)

    ax.set_title("Pixel Intensity Distribution per Class",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Pixel intensity (0 = black, 255 = white)",
                  fontsize=12, labelpad=10)
    ax.set_ylabel("Normalized frequency", fontsize=12, labelpad=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(BASE / "eda_pixel_intensities.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    saved -> eda_pixel_intensities.png")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  EDA — Exploratory Data Analysis")
    print("=" * 60)

    data = collect_dataset()
    print(f"\nFound {len(data)} class(es), "
          f"{sum(len(v) for v in data.values())} image(s) total.\n")

    chart_class_distribution(data)
    chart_sample_grid(data)
    chart_image_dimensions(data)
    chart_pixel_intensities(data)

    print("\n" + "=" * 60)
    print("  EDA complete — 4 PNG files generated.")
    print("  Drop them into the report's Figure 2 / EDA section.")
    print("=" * 60)
