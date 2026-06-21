"""
═══════════════════════════════════════════════════════════════════════
  hypertune.py — Hyperparameter Optimization (GridSearchCV)
═══════════════════════════════════════════════════════════════════════

Performs a grid search over the SVM and PCA hyperparameters using
5-fold cross-validation. Saves three outputs for the report:

  1. hyperparameter_results.csv — full grid results sorted by mean CV score
  2. best_params.txt            — the winning hyperparameter combination
  3. top_10_configs.png         — bar chart of the top-10 configurations

Run from inside the Site2_Attendance folder:
    python hypertune.py

Grid size: 4 (C) × 4 (gamma) × 3 (n_components) = 48 combinations
With 5-fold CV that's 240 model fits — takes 1–3 minutes on a laptop.
═══════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# ════════════════════════════════════════════════════════════════════
#  CONFIG (must match app.py exactly)
# ════════════════════════════════════════════════════════════════════
BASE     = Path(__file__).parent
DATASET  = BASE / "dataset"
ALLOWED  = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE = (100, 100)


# ════════════════════════════════════════════════════════════════════
#  Preprocessing helpers (same as app.py + evaluate.py)
# ════════════════════════════════════════════════════════════════════
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


def ensure_3channel(img):
    if img is None: return None
    if len(img.shape) == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:   return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:   return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def detect_faces(bgr):
    bgr = ensure_3channel(bgr)
    if bgr is None: return []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    res = face_detector.process(rgb)
    out = []
    if res.detections:
        for det in res.detections:
            bbox = det.location_data.relative_bounding_box
            x  = max(0, int(bbox.xmin * w));  y  = max(0, int(bbox.ymin * h))
            bw = min(int(bbox.width  * w), w - x)
            bh = min(int(bbox.height * h), h - y)
            if bw <= 0 or bh <= 0: continue
            kps = det.location_data.relative_keypoints
            out.append({
                "box": (x, y, bw, bh),
                "right_eye": (int(kps[0].x * w), int(kps[0].y * h)),
                "left_eye":  (int(kps[1].x * w), int(kps[1].y * h)),
                "nose":      (int(kps[2].x * w), int(kps[2].y * h)),
            })
    return out


def align_face(bgr, face, output_size=IMG_SIZE):
    out_w, out_h = output_size
    dst = np.array([
        (int(out_w * 0.35), int(out_h * 0.40)),
        (int(out_w * 0.65), int(out_h * 0.40)),
        (int(out_w * 0.50), int(out_h * 0.65))
    ], dtype=np.float32)
    src = np.array([face["right_eye"], face["left_eye"], face["nose"]],
                   dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    aligned = cv2.warpAffine(bgr, M, (out_w, out_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if aligned is None or aligned.size == 0: return None
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)


def fallback_resize(bgr, size=IMG_SIZE):
    bgr = ensure_3channel(bgr)
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), size)


def looks_like_tight_crop(bgr):
    if bgr is None: return False
    h, w = bgr.shape[:2]
    if min(h, w) < 50: return False
    return 0.6 < (w / max(h, 1)) < 1.6


# ════════════════════════════════════════════════════════════════════
#  Build dataset
# ════════════════════════════════════════════════════════════════════
def build_dataset():
    print("[1/3] Building dataset (running detection + alignment)...")
    X, y_names = [], []
    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir(): continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in ALLOWED: continue
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            bgr = ensure_3channel(bgr)
            faces = detect_faces(bgr)
            if faces:
                best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                aligned = align_face(bgr, best)
                if aligned is None: continue
                vec = aligned.flatten().astype(np.float64)
            elif looks_like_tight_crop(bgr):
                vec = fallback_resize(bgr).flatten().astype(np.float64)
            else:
                continue
            X.append(vec)
            y_names.append(person_dir.name)
    X = np.array(X, dtype=np.float64)
    le = LabelEncoder()
    y = le.fit_transform(y_names)
    print(f"    {len(X)} samples · {len(le.classes_)} classes\n")
    return X, y, le


# ════════════════════════════════════════════════════════════════════
#  Grid search
# ════════════════════════════════════════════════════════════════════
def run_grid_search(X, y, le):
    print("[2/3] Running GridSearchCV (5-fold, 48 combinations, ~1-3 min)...\n")

    n_classes = len(le.classes_)
    n_lda     = max(1, n_classes - 1)

    pipeline = Pipeline([
        ("pca", PCA(whiten=True, random_state=42)),
        ("lda", LinearDiscriminantAnalysis(n_components=n_lda)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    # 4 × 4 × 3 = 48 combinations
    param_grid = {
        "pca__n_components": [30, 50, 70],
        "svm__C":            [0.1, 1, 10, 100],
        "svm__gamma":        ["scale", "auto", 0.01, 0.1],
    }

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=5, scoring="accuracy",
        n_jobs=-1, verbose=1, return_train_score=True
    )
    grid.fit(X, y)

    print(f"\n    Best CV accuracy: {grid.best_score_ * 100:.2f}%")
    print(f"    Best parameters:  {grid.best_params_}")
    return grid


# ════════════════════════════════════════════════════════════════════
#  Save outputs
# ════════════════════════════════════════════════════════════════════
def save_outputs(grid):
    print("\n[3/3] Saving outputs...")

    # ── CSV with full grid results ──
    df = pd.DataFrame(grid.cv_results_)
    keep = ["param_pca__n_components", "param_svm__C", "param_svm__gamma",
            "mean_test_score", "std_test_score",
            "mean_train_score", "rank_test_score"]
    df = df[keep].sort_values("rank_test_score").reset_index(drop=True)
    df.columns = ["PCA components", "SVM C", "SVM gamma",
                  "CV accuracy (mean)", "CV accuracy (std)",
                  "Train accuracy (mean)", "Rank"]
    df["CV accuracy (mean)"]    = (df["CV accuracy (mean)"] * 100).round(2)
    df["CV accuracy (std)"]     = (df["CV accuracy (std)"]  * 100).round(2)
    df["Train accuracy (mean)"] = (df["Train accuracy (mean)"] * 100).round(2)
    df.to_csv(BASE / "hyperparameter_results.csv", index=False)
    print(f"    saved -> hyperparameter_results.csv  ({len(df)} configs)")

    # ── Best params text ──
    with open(BASE / "best_params.txt", "w") as f:
        f.write("BEST HYPERPARAMETER CONFIGURATION\n")
        f.write("=" * 50 + "\n\n")
        for k, v in grid.best_params_.items():
            f.write(f"  {k:25s} = {v}\n")
        f.write(f"\n  Best CV accuracy        = {grid.best_score_ * 100:.2f}%\n")
        f.write(f"  Total configs tested    = {len(grid.cv_results_['params'])}\n")
        f.write(f"  Cross-validation folds  = 5\n")
    print("    saved -> best_params.txt")

    # ── Top-10 chart ──
    top10 = df.head(10).copy()
    labels = [f"PCA={r['PCA components']}, C={r['SVM C']}, γ={r['SVM gamma']}"
              for _, r in top10.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120)
    bars = ax.barh(range(len(top10)), top10["CV accuracy (mean)"],
                   color="#6ee7b7", edgecolor="#1f2937", linewidth=1.2)
    for i, (bar, val, std) in enumerate(zip(bars,
                                            top10["CV accuracy (mean)"],
                                            top10["CV accuracy (std)"])):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}% ± {std:.1f}", va="center", fontsize=10,
                fontweight="bold")

    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV accuracy (%)", fontsize=12, labelpad=10)
    ax.set_title("Top-10 Hyperparameter Configurations",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(BASE / "top_10_configs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    saved -> top_10_configs.png")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Hyperparameter Optimization")
    print("=" * 60 + "\n")
    X, y, le = build_dataset()
    grid = run_grid_search(X, y, le)
    save_outputs(grid)
    print("\n" + "=" * 60)
    print("  Done — 3 outputs generated:")
    print("    • hyperparameter_results.csv")
    print("    • best_params.txt")
    print("    • top_10_configs.png")
    print("=" * 60)
