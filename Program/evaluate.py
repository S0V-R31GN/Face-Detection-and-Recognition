"""
═══════════════════════════════════════════════════════════════════════
  evaluate.py — Honest Model Evaluation with Visualizations
═══════════════════════════════════════════════════════════════════════

Re-runs the same train/test split as app.py (random_state=42), evaluates
the trained model on the held-out test set, and produces:

  1. confusion_matrix.png       — heatmap with seaborn (per-class accuracy)
  2. classification_report.txt  — per-class precision / recall / F1 / support
  3. eigenfaces.png             — first 16 PCA components visualized

Run from inside the Site2_Attendance folder (where model.pkl + dataset/ live):
    python evaluate.py
═══════════════════════════════════════════════════════════════════════
"""

import os, pickle
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ════════════════════════════════════════════════════════════════════
#  CONFIG — must match app.py exactly
# ════════════════════════════════════════════════════════════════════
BASE       = Path(__file__).parent
DATASET    = BASE / "dataset"
MODEL_FILE = BASE / "model.pkl"
ALLOWED    = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE   = (100, 100)
PALETTE    = ["#6ee7b7", "#818cf8", "#fbbf24", "#f87171"]


# ════════════════════════════════════════════════════════════════════
#  Preprocessing — same logic as app.py
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
    src = np.array([face["right_eye"], face["left_eye"], face["nose"]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    aligned = cv2.warpAffine(bgr, M, (out_w, out_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
#  Load dataset
# ════════════════════════════════════════════════════════════════════
def build_dataset():
    print("[1/4] Building dataset (running detection + alignment)...")
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
    return np.array(X, dtype=np.float64), np.array(y_names)


# ════════════════════════════════════════════════════════════════════
#  Load model
# ════════════════════════════════════════════════════════════════════
def load_model():
    print("[2/4] Loading trained model...")
    if not MODEL_FILE.exists():
        raise SystemExit("[ERROR] model.pkl not found — train the model first.")
    with open(MODEL_FILE, "rb") as f:
        s = pickle.load(f)
    return s["pca"], s.get("lda"), s["svm"], s["le"]


# ════════════════════════════════════════════════════════════════════
#  Evaluate
# ════════════════════════════════════════════════════════════════════
def evaluate(X, y_names, pca, lda, svm, le):
    print("[3/4] Re-running 80/20 stratified split (random_state=42)...")
    y = le.transform(y_names)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_test_pca = pca.transform(X_test)
    feats = lda.transform(X_test_pca) if lda is not None else X_test_pca
    y_pred = svm.predict(feats)

    acc = accuracy_score(y_test, y_pred) * 100
    print(f"    Test accuracy on {len(y_test)} held-out samples: {acc:.2f}%\n")

    # ── 1. Confusion matrix ──
    print("[4/4] Generating evaluation visualizations...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={"label": "Count"},
                annot_kws={"size": 14, "fontweight": "bold"},
                linewidths=1, linecolor="white", ax=ax)
    ax.set_title(f"Confusion Matrix — Test Set (accuracy = {acc:.1f}%)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted class", fontsize=12, labelpad=10)
    ax.set_ylabel("True class", fontsize=12, labelpad=10)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(BASE / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    saved -> confusion_matrix.png")

    # ── 2. Classification report ──
    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_, digits=3)
    print("\n" + "─" * 60)
    print("CLASSIFICATION REPORT")
    print("─" * 60)
    print(report)
    print("─" * 60)
    with open(BASE / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.2f}%\n")
        f.write(f"Test samples:  {len(y_test)}\n\n")
        f.write(report)
    print("    saved -> classification_report.txt")

    # ── 3. Eigenfaces visualization ──
    print("\n    Visualizing eigenfaces (first 16 PCA components)...")
    eigenfaces = pca.components_[:16].reshape(-1, *IMG_SIZE)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), dpi=120)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(eigenfaces[i], cmap="gray")
        ax.set_title(f"Eigenface #{i+1}", fontsize=10)
        ax.axis("off")
    plt.suptitle("First 16 PCA Components (Eigenfaces)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(BASE / "eigenfaces.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    saved -> eigenfaces.png")


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Model Evaluation")
    print("=" * 60 + "\n")

    X, y_names = build_dataset()
    pca, lda, svm, le = load_model()
    evaluate(X, y_names, pca, lda, svm, le)

    print("\n" + "=" * 60)
    print("  Evaluation complete — 3 outputs generated:")
    print("    • confusion_matrix.png")
    print("    • classification_report.txt")
    print("    • eigenfaces.png")
    print("=" * 60)
