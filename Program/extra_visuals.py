"""
═══════════════════════════════════════════════════════════════════════
  extra_visuals.py — Generates the 5 missing report figures
═══════════════════════════════════════════════════════════════════════

Produces:
  1. lda_clusters.png         — 2D scatter of LDA-projected training faces
                                 (the clustering visualization)
  2. pca_variance_curve.png   — cumulative explained variance vs n_components
                                 (justifies "why 50")
  3. mean_face.png            — the average face from training data
  4. alignment_demo.png       — before/after alignment on a sample image
  5. pipeline_demo.png        — one face going through every pipeline step

Run from inside the Site2_Attendance folder (where model.pkl + dataset/ live):
    python extra_visuals.py
═══════════════════════════════════════════════════════════════════════
"""

import os, pickle
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mediapipe as mp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# ── Config ──
BASE       = Path(__file__).parent
DATASET    = BASE / "dataset"
MODEL_FILE = BASE / "model.pkl"
ALLOWED    = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE   = (100, 100)
PALETTE    = ["#6ee7b7", "#818cf8", "#fbbf24", "#f87171", "#34d399", "#60a5fa"]

# ── MediaPipe ──
mp_fd = mp.solutions.face_detection
detector = mp_fd.FaceDetection(min_detection_confidence=0.5, model_selection=0)


def ensure_3channel(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def detect_kps(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    results = detector.process(rgb)
    if not results.detections:
        return None
    det = results.detections[0]
    kps = det.location_data.relative_keypoints
    return {
        "right_eye": (int(kps[0].x * w), int(kps[0].y * h)),
        "left_eye":  (int(kps[1].x * w), int(kps[1].y * h)),
        "nose":      (int(kps[2].x * w), int(kps[2].y * h)),
    }


def align_face(bgr, kps, size=IMG_SIZE):
    out_w, out_h = size
    dst = np.array([
        (int(out_w * 0.35), int(out_h * 0.40)),
        (int(out_w * 0.65), int(out_h * 0.40)),
        (int(out_w * 0.50), int(out_h * 0.65)),
    ], dtype=np.float32)
    src = np.array([kps["right_eye"], kps["left_eye"], kps["nose"]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    aligned = cv2.warpAffine(bgr, M, (out_w, out_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)


def load_dataset():
    """Returns (X, y, label_names) — X is flattened aligned faces."""
    X, y, labels = [], [], []
    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        labels.append(name)
        idx = len(labels) - 1
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in ALLOWED:
                continue
            img = cv2.imread(str(img_path))
            img = ensure_3channel(img)
            if img is None:
                continue
            kps = detect_kps(img)
            if kps is None:
                # fallback: resize whole image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                aligned = cv2.resize(gray, IMG_SIZE)
            else:
                aligned = align_face(img, kps)
            X.append(aligned.flatten().astype(np.float64))
            y.append(idx)
    return np.array(X), np.array(y), labels


# ════════════════════════════════════════════════════════════════════
#  1. LDA 2D CLUSTERS  — the headline visual
# ════════════════════════════════════════════════════════════════════
def fig_lda_clusters(X_train, y_train, labels):
    print("[1/5] Building LDA cluster scatter...")
    # Match your pipeline: PCA 50 → LDA
    pca = PCA(n_components=50, whiten=True, random_state=42).fit(X_train)
    X_pca = pca.transform(X_train)
    lda = LinearDiscriminantAnalysis(n_components=2).fit(X_pca, y_train)
    X_lda = lda.transform(X_pca)

    fig, ax = plt.subplots(figsize=(11, 8))
    for i, name in enumerate(labels):
        mask = y_train == i
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                   c=PALETTE[i % len(PALETTE)],
                   label=name, s=90, alpha=0.75,
                   edgecolors="black", linewidths=0.8)
        # cluster center
        cx, cy = X_lda[mask, 0].mean(), X_lda[mask, 1].mean()
        ax.scatter(cx, cy, marker="X", c="black", s=240, zorder=5)
        ax.annotate(name, (cx, cy), xytext=(8, 8),
                    textcoords="offset points", fontsize=11, fontweight="bold")

    ax.set_title("LDA Projection — Each Person Forms a Distinct Cluster",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("LDA Component 1", fontsize=12)
    ax.set_ylabel("LDA Component 2", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("lda_clusters.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("    ✓ lda_clusters.png")


# ════════════════════════════════════════════════════════════════════
#  2. PCA CUMULATIVE VARIANCE CURVE — justifies "why 50"
# ════════════════════════════════════════════════════════════════════
def fig_variance_curve(X_train):
    print("[2/5] Building PCA variance curve...")
    pca = PCA(n_components=min(150, X_train.shape[0] - 1),
              random_state=42).fit(X_train)
    cum = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(range(1, len(cum) + 1), cum, color="#6366f1",
            linewidth=2.5, marker="o", markersize=4)
    ax.axvline(50, color="#ef4444", linestyle="--", linewidth=2,
               label=f"Our choice: 50 components → {cum[49]:.1f}% variance")
    ax.axhline(cum[49], color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.fill_between(range(1, len(cum) + 1), 0, cum, alpha=0.15, color="#6366f1")

    ax.set_title("PCA Cumulative Explained Variance",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Number of PCA components", fontsize=12)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=12)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, len(cum))
    ax.set_ylim(0, 102)
    plt.tight_layout()
    plt.savefig("pca_variance_curve.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    ✓ pca_variance_curve.png  (50 comp = {cum[49]:.1f}% var)")


# ════════════════════════════════════════════════════════════════════
#  3. MEAN FACE — average of all training samples
# ════════════════════════════════════════════════════════════════════
def fig_mean_face(X_train):
    print("[3/5] Building mean face...")
    mean = X_train.mean(axis=0).reshape(IMG_SIZE)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mean, cmap="gray")
    ax.set_title("Mean Face\n(PCA subtracts this before projection)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("mean_face.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("    ✓ mean_face.png")


# ════════════════════════════════════════════════════════════════════
#  4. ALIGNMENT DEMO — before / after on a real image
# ════════════════════════════════════════════════════════════════════
def fig_alignment_demo():
    print("[4/5] Building alignment demo...")
    # Find first image with a detectable face
    sample = None
    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in ALLOWED:
                continue
            img = cv2.imread(str(img_path))
            img = ensure_3channel(img)
            if img is None:
                continue
            kps = detect_kps(img)
            if kps is not None:
                sample = (img, kps, person_dir.name)
                break
        if sample:
            break

    if not sample:
        print("    ✗ No alignable image found, skipping")
        return

    img, kps, name = sample
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # draw keypoints on left
    annotated = rgb.copy()
    for label, pt, color in [
        ("R-eye", kps["right_eye"], (255, 0, 0)),
        ("L-eye", kps["left_eye"],  (0, 0, 255)),
        ("Nose",  kps["nose"],      (0, 255, 0)),
    ]:
        cv2.circle(annotated, pt, 8, color, -1)
        cv2.circle(annotated, pt, 12, (255, 255, 255), 2)

    aligned = align_face(img, kps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(annotated)
    axes[0].set_title("BEFORE — Raw image with MediaPipe keypoints\n"
                      "(red = R-eye, blue = L-eye, green = nose)",
                      fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(aligned, cmap="gray")
    axes[1].set_title("AFTER — Affine-warped 100×100 grayscale\n"
                      "Eyes at fixed positions (35,40) and (65,40)",
                      fontsize=13, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("Face Alignment Pipeline", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("alignment_demo.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("    ✓ alignment_demo.png")


# ════════════════════════════════════════════════════════════════════
#  5. FULL PIPELINE DEMO — one face through every stage
# ════════════════════════════════════════════════════════════════════
def fig_pipeline_demo(X_train, y_train, labels):
    print("[5/5] Building full pipeline demo...")
    # Find one sample image
    sample = None
    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in ALLOWED:
                continue
            img = cv2.imread(str(img_path))
            img = ensure_3channel(img)
            if img is None:
                continue
            kps = detect_kps(img)
            if kps is not None:
                sample = (img, kps)
                break
        if sample:
            break

    if not sample:
        print("    ✗ No alignable image found, skipping")
        return

    img, kps = sample

    # Stage 1: original
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Stage 2: with detection + keypoints
    annotated = rgb.copy()
    for pt, color in [
        (kps["right_eye"], (255, 0, 0)),
        (kps["left_eye"], (0, 0, 255)),
        (kps["nose"], (0, 255, 0)),
    ]:
        cv2.circle(annotated, pt, 10, color, -1)

    # Stage 3: aligned grayscale
    aligned = align_face(img, kps)

    # Stage 4: PCA reconstruction (visualize what model sees in 50D)
    pca = PCA(n_components=50, whiten=True, random_state=42).fit(X_train)
    flat = aligned.flatten().astype(np.float64).reshape(1, -1)
    coeffs = pca.transform(flat)
    recon = pca.inverse_transform(coeffs).reshape(IMG_SIZE)

    # Stage 5: LDA position (text annotation)
    lda = LinearDiscriminantAnalysis(n_components=min(5, len(labels) - 1)).fit(
        pca.transform(X_train), y_train)
    lda_coords = lda.transform(coeffs)[0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("1. Raw input\n(BGR, variable size)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(annotated)
    axes[1].set_title("2. MediaPipe detection\n(6 keypoints → use 3)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(aligned, cmap="gray")
    axes[2].set_title("3. Aligned + grayscale\n(100×100 = 10,000 pixels)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(recon, cmap="gray")
    axes[3].set_title(f"4. PCA reconstruction\n(50 components — what model sees)",
                      fontsize=12, fontweight="bold")
    axes[3].axis("off")

    plt.suptitle("Pipeline Visualization on One Real Face",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("pipeline_demo.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    ✓ pipeline_demo.png  (LDA coords: {lda_coords[:2].round(2).tolist()})")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Extra Visualizations for Report")
    print("=" * 60)

    print("\nLoading dataset...")
    X, y, labels = load_dataset()
    print(f"  Loaded {len(X)} samples across {len(labels)} classes")
    print(f"  Classes: {labels}")

    # Same split as app.py / evaluate.py
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Training set: {len(X_train)} samples\n")

    fig_lda_clusters(X_train, y_train, labels)
    fig_variance_curve(X_train)
    fig_mean_face(X_train)
    fig_alignment_demo()
    fig_pipeline_demo(X_train, y_train, labels)

    print("\n" + "=" * 60)
    print("  Done. 5 new PNGs in this folder. Send them all to Claude.")
    print("=" * 60)


if __name__ == "__main__":
    main()
