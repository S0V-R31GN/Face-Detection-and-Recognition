"""
═══════════════════════════════════════════════════════════════════════
  SITE 2 — ATTENDANCE SYSTEM (BACKEND, PATCHED)
═══════════════════════════════════════════════════════════════════════

PATCH NOTES (over previous version):
  Fix 1 — Convert grayscale → 3-channel BGR before MediaPipe
          (so old grayscale dataset photos are detected/recognized)
  Fix 2 — MediaPipe model_selection=0 (short range)
          (you no longer have to lean back to be detected)
  Fix 3 — Honest train/test split — reports both train & test accuracy
          so you can see real generalization, not memorized accuracy
  Fix 4 — Prediction-time fallback for tightly cropped images
          (matches training-time behavior for old grayscale crops)
  Fix 5 — Returns box coordinates per face so the frontend can
          draw boxes ON TOP of live video (no more "freeze" feeling)
═══════════════════════════════════════════════════════════════════════
"""

import os, pickle, base64, uuid, shutil, sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, send_from_directory
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════
BASE          = Path(__file__).parent
DATASET       = BASE / "dataset"
MODEL_FILE    = BASE / "model.pkl"
DB_PATH       = BASE / "attendance.db"
ALLOWED       = {".jpg", ".jpeg", ".png", ".webp"}
IMG_SIZE      = (100, 100)
N_COMPONENTS  = 50
MODEL_VERSION = 2

DATASET.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")


# ════════════════════════════════════════════════════════════════════
#  MEDIAPIPE FACE DETECTOR
# ════════════════════════════════════════════════════════════════════
# FIX 2: model_selection=0 → "short range" — best for webcam users
#         sitting at a normal distance (within ~2 metres of camera).
#         Previously model_selection=1 ("full range") often missed
#         close-up faces.
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# ════════════════════════════════════════════════════════════════════
#  GLOBAL MODEL STATE
# ════════════════════════════════════════════════════════════════════
pca          = None
lda          = None
svm          = None
le           = None
model_loaded = False
model_meta   = {}


# ════════════════════════════════════════════════════════════════════
#  DATABASE — Attendance Logging
# ════════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT,
            date       TEXT,
            time       TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()


def mark_attendance(name, confidence):
    if name == "Unknown":
        return "N/A"
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")
    c.execute("SELECT 1 FROM logs WHERE name=? AND date=?", (name, today))
    if c.fetchone():
        conn.close()
        return "Already Present"
    c.execute(
        "INSERT INTO logs (name, date, time, confidence) VALUES (?, ?, ?, ?)",
        (name, today, now, confidence)
    )
    conn.commit()
    conn.close()
    return "Logged"


init_db()


# ════════════════════════════════════════════════════════════════════
#  IMAGE UTILITIES
# ════════════════════════════════════════════════════════════════════

def allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED


def ensure_3channel(img):
    """
    FIX 1 — MediaPipe requires a 3-channel BGR image. Some older
    images load as single-channel grayscale or 4-channel BGRA.
    This converts everything to a clean 3-channel BGR.

    Without this fix, grayscale photos (e.g. old dataset captures
    saved as grayscale .jpg) silently failed face detection.
    """
    if img is None:
        return None
    if len(img.shape) == 2:                       # 1-channel grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:                         # 1-channel with shape (h,w,1)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                         # 4-channel BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def decode_image(file_storage):
    data = file_storage.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return ensure_3channel(img)


def decode_base64_frame(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return ensure_3channel(img)


# ════════════════════════════════════════════════════════════════════
#  FACE DETECTION + ALIGNMENT
# ════════════════════════════════════════════════════════════════════

def detect_faces(bgr):
    """
    Detect faces using MediaPipe.
    Returns a list of dicts: [{box, right_eye, left_eye, nose, score}, ...]
    """
    bgr = ensure_3channel(bgr)        # FIX 1 — defensive
    if bgr is None:
        return []

    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    results = face_detector.process(rgb)

    out = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x  = max(0, int(bbox.xmin * w))
            y  = max(0, int(bbox.ymin * h))
            bw = min(int(bbox.width  * w), w - x)
            bh = min(int(bbox.height * h), h - y)
            if bw <= 0 or bh <= 0:
                continue
            kps = det.location_data.relative_keypoints
            out.append({
                "box":       (x, y, bw, bh),
                "right_eye": (int(kps[0].x * w), int(kps[0].y * h)),
                "left_eye":  (int(kps[1].x * w), int(kps[1].y * h)),
                "nose":      (int(kps[2].x * w), int(kps[2].y * h)),
                "score":     float(det.score[0]),
            })
    return out


def align_face(bgr, face, output_size=IMG_SIZE):
    """
    Affine-warp a face so eyes & nose land at canonical positions.
    See previous version comments for full explanation.
    """
    out_w, out_h = output_size
    dst_right_eye = (int(out_w * 0.35), int(out_h * 0.40))
    dst_left_eye  = (int(out_w * 0.65), int(out_h * 0.40))
    dst_nose      = (int(out_w * 0.50), int(out_h * 0.65))

    src_pts = np.array([face["right_eye"], face["left_eye"], face["nose"]],
                       dtype=np.float32)
    dst_pts = np.array([dst_right_eye, dst_left_eye, dst_nose],
                       dtype=np.float32)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    aligned = cv2.warpAffine(bgr, M, (out_w, out_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    if aligned is None or aligned.size == 0:
        return None
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)


def fallback_resize(bgr, output_size=IMG_SIZE):
    """For pre-cropped face images (old grayscale dataset photos)."""
    bgr  = ensure_3channel(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, output_size)


def looks_like_tight_crop(bgr):
    """
    Heuristic: if MediaPipe found nothing, is this image
    probably already a tightly-cropped face (no surrounding context)?
    Old dataset photos saved by collect_faces.py were 200x200 grayscale
    crops — those are the ones that need this fallback.
    """
    if bgr is None:
        return False
    h, w = bgr.shape[:2]
    if min(h, w) < 50:
        return False                # too tiny to be a usable face
    ratio = w / max(h, 1)
    return 0.6 < ratio < 1.6       # roughly square


# ════════════════════════════════════════════════════════════════════
#  ANNOTATION (still used for upload preview)
# ════════════════════════════════════════════════════════════════════

def annotate_image(bgr, faces, results):
    out = bgr.copy()
    for face, res in zip(faces, results):
        x, y, w, h = face["box"]
        is_known = res["name"] != "Unknown"
        color    = (34, 197, 94) if is_known else (239, 68, 68)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        text = f'{res["name"]}  {res["confidence"]:.1f}%'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x, y - th - 14), (x + tw + 10, y), color, -1)
        cv2.putText(out, text, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ════════════════════════════════════════════════════════════════════
#  MODEL LOAD / SAVE
# ════════════════════════════════════════════════════════════════════

def load_model():
    global pca, lda, svm, le, model_loaded, model_meta
    if not MODEL_FILE.exists():
        model_loaded = False
        return
    with open(MODEL_FILE, "rb") as f:
        state = pickle.load(f)
    pca = state["pca"]
    svm = state["svm"]
    le  = state["le"]
    lda = state.get("lda")
    model_meta = {
        "version":            state.get("version", 1),
        "n_components_pca":   int(getattr(pca, "n_components_", 0)),
        "n_components_lda":   int(getattr(lda, "n_components_", 0)) if lda is not None else 0,
        "explained_variance": round(float(pca.explained_variance_ratio_.sum()) * 100, 2),
        "classes":            list(le.classes_) if le is not None else [],
    }
    model_loaded = True


def save_model():
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({
            "pca": pca, "lda": lda, "svm": svm, "le": le,
            "version": MODEL_VERSION, "img_size": IMG_SIZE,
        }, f)


# ════════════════════════════════════════════════════════════════════
#  TRAINING — PCA → LDA → SVM with HONEST train/test split
# ════════════════════════════════════════════════════════════════════

def train_model():
    """
    FIX 3 — Splits dataset 80/20 train/test before fitting.
    Reports both train accuracy AND test accuracy.

    Why this matters academically:
      - Train accuracy alone is meaningless (a model can perfectly
        memorize 200 photos and fail on every new one)
      - Test accuracy estimates real-world generalization
      - The doctor will expect to see this distinction; it's standard
        ML methodology covered in your course (Ch. 12).
    """
    global pca, lda, svm, le, model_loaded

    X, y_names, skipped = [], [], []

    for person_dir in sorted(DATASET.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in ALLOWED:
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            bgr = ensure_3channel(bgr)        # FIX 1

            faces = detect_faces(bgr)
            if faces:
                best = max(faces, key=lambda f: f["box"][2] * f["box"][3])
                aligned = align_face(bgr, best)
                if aligned is None:
                    skipped.append(img_path.name); continue
                vec = aligned.flatten().astype(np.float64)
            elif looks_like_tight_crop(bgr):
                vec = fallback_resize(bgr).flatten().astype(np.float64)
            else:
                skipped.append(img_path.name); continue

            X.append(vec)
            y_names.append(person_dir.name)

    if len(X) < 2:
        return {"ok": False, "error": "Need at least 2 valid images.",
                "skipped": skipped[:10]}
    if len(set(y_names)) < 2:
        return {"ok": False, "error": "Need at least 2 different persons."}

    X = np.array(X, dtype=np.float64)
    _le = LabelEncoder()
    y   = _le.fit_transform(y_names)
    n_classes = len(_le.classes_)

    # ── FIX 3: Train/Test split if every class has ≥5 samples ──
    _, counts = np.unique(y, return_counts=True)
    can_split = counts.min() >= 5

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_test, y_test   = None, None

    # ── PCA on train data only (no test leakage) ──
    n_pca = min(N_COMPONENTS, len(X_train) - 1, X_train.shape[1])
    _pca  = PCA(n_components=n_pca, whiten=True, random_state=42)
    X_train_pca = _pca.fit_transform(X_train)

    # ── LDA ──
    n_lda = max(1, min(n_classes - 1, n_pca))
    _lda  = LinearDiscriminantAnalysis(n_components=n_lda)
    X_train_lda = _lda.fit_transform(X_train_pca, y_train)

    # ── SVM ──
    _svm = SVC(kernel="rbf", C=1.0, gamma="scale",
               probability=True, random_state=42)
    _svm.fit(X_train_lda, y_train)

    train_acc = accuracy_score(y_train, _svm.predict(X_train_lda)) * 100

    test_acc = None
    if X_test is not None:
        X_test_pca = _pca.transform(X_test)
        X_test_lda = _lda.transform(X_test_pca)
        test_acc   = accuracy_score(y_test, _svm.predict(X_test_lda)) * 100

    explained = float(_pca.explained_variance_ratio_.sum()) * 100

    pca, lda, svm, le = _pca, _lda, _svm, _le
    save_model()
    load_model()

    msg_test = f" · test={test_acc:.1f}%" if test_acc is not None else " · (too few samples per class for test split)"
    return {
        "ok":                   True,
        "persons":              list(_le.classes_),
        "samples":              len(X),
        "train_samples":        len(X_train),
        "test_samples":         len(X_test) if X_test is not None else 0,
        "n_components_pca":     n_pca,
        "n_components_lda":     n_lda,
        "explained_variance":   round(explained, 2),
        "train_accuracy":       round(train_acc, 2),
        "test_accuracy":        round(test_acc, 2) if test_acc is not None else None,
        "skipped":              skipped[:10],
        "message": (
            f"PCA+LDA+SVM trained | {len(X)} samples · {n_classes} persons | "
            f"PCA={n_pca} · LDA={n_lda} · variance={explained:.1f}% | "
            f"train={train_acc:.1f}%{msg_test}"
        )
    }


# ════════════════════════════════════════════════════════════════════
#  PREDICTION
# ════════════════════════════════════════════════════════════════════

def predict_face(vec, threshold=40.0):
    if not model_loaded or pca is None:
        return {"name": "Unknown", "confidence": 0.0}
    feats = pca.transform(vec.reshape(1, -1))
    if lda is not None:
        feats = lda.transform(feats)
    probs       = svm.predict_proba(feats)[0]
    best_idx    = int(np.argmax(probs))
    confidence  = float(probs[best_idx]) * 100.0
    if confidence < threshold:
        return {"name": "Unknown", "confidence": round(confidence, 1)}
    return {
        "name":       le.inverse_transform([best_idx])[0],
        "confidence": round(confidence, 1)
    }


# ════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/status")
def api_status():
    persons, counts = [], {}
    for d in sorted(DATASET.iterdir()):
        if d.is_dir():
            persons.append(d.name)
            counts[d.name] = sum(1 for f in d.iterdir() if f.suffix.lower() in ALLOWED)
    return jsonify({
        "model_trained": model_loaded,
        "persons":       persons,
        "image_counts":  counts,
        "model_info":    model_meta,
        "algorithm":     "MediaPipe → Align → PCA → LDA → SVM (RBF)"
    })


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    Mode A (webcam):  send `frame` as base64  + log_attendance=true
    Mode B (upload):  send `image` multipart   + log_attendance=false

    FIX 4 — If MediaPipe finds no faces but the image looks like a
            tight face crop (old grayscale dataset format), we still
            try to recognize using the whole-image fallback. This
            mirrors what training does, so old grayscale photos
            uploaded for testing now work.

    FIX 5 — Each result now includes "box": [x, y, w, h] so the
            frontend can draw boxes on a canvas overlaid on the
            live video, instead of having to replace the video
            with a server-annotated image (which felt frozen).
    """
    if "image" in request.files:
        bgr = decode_image(request.files["image"])
    elif request.form.get("frame"):
        bgr = decode_base64_frame(request.form["frame"])
    elif request.is_json and request.json.get("frame"):
        bgr = decode_base64_frame(request.json["frame"])
    else:
        return jsonify({"ok": False, "error": "No image/frame provided"}), 400

    if bgr is None:
        return jsonify({"ok": False, "error": "Could not decode image"}), 400

    threshold      = float(request.values.get("threshold", 40))
    log_attendance = str(request.values.get("log_attendance", "false")).lower() == "true"

    faces   = detect_faces(bgr)
    results = []

    if faces:
        # Normal path — MediaPipe found faces
        for f in faces:
            aligned = align_face(bgr, f)
            if aligned is None:
                continue
            vec = aligned.flatten().astype(np.float64)
            res = predict_face(vec, threshold)
            x, y, w, h = f["box"]
            res["box"] = [int(x), int(y), int(w), int(h)]    # FIX 5

            if log_attendance and res["name"] != "Unknown":
                res["attendance"] = mark_attendance(res["name"], res["confidence"])
            else:
                res["attendance"] = "N/A"
            results.append(res)

    elif looks_like_tight_crop(bgr):
        # FIX 4 — fallback for tight pre-cropped face images
        h, w = bgr.shape[:2]
        vec = fallback_resize(bgr).flatten().astype(np.float64)
        res = predict_face(vec, threshold)
        res["box"] = [0, 0, int(w), int(h)]                  # FIX 5

        if log_attendance and res["name"] != "Unknown":
            res["attendance"] = mark_attendance(res["name"], res["confidence"])
        else:
            res["attendance"] = "N/A"
        results.append(res)
        # Synthesize a box for the annotator
        faces = [{"box": (0, 0, w, h), "right_eye": (0, 0),
                  "left_eye": (0, 0), "nose": (0, 0)}]

    annotated = annotate_image(bgr, faces, results) if (faces and results) else None

    return jsonify({
        "ok":         True,
        "faces":      len(results),
        "results":    results,
        "annotated":  annotated,
        "message":    f"Detected {len(results)} face(s)"
    })


@app.route("/api/upload_dataset", methods=["POST"])
def api_upload_dataset():
    name = request.form.get("person_name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "person_name is required"}), 400
    files = request.files.getlist("images")
    if not files:
        return jsonify({"ok": False, "error": "No images uploaded"}), 400

    person_dir = DATASET / name
    person_dir.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []

    for f in files:
        if not allowed(f.filename):
            skipped.append(f.filename + " (unsupported)"); continue
        bgr = decode_image(f)
        if bgr is None:
            skipped.append(f.filename + " (decode error)"); continue

        # Accept if MediaPipe detects OR if it looks like a pre-cropped face
        faces = detect_faces(bgr)
        if not faces and not looks_like_tight_crop(bgr):
            skipped.append(f.filename + " (no face detected)"); continue

        fname = f"{uuid.uuid4().hex}{Path(f.filename).suffix.lower()}"
        cv2.imwrite(str(person_dir / fname), bgr)
        saved.append(fname)

    return jsonify({
        "ok":      len(saved) > 0,
        "saved":   len(saved),
        "skipped": skipped,
        "person":  name,
        "message": f"Saved {len(saved)} image(s) for '{name}'"
    })


@app.route("/api/capture_to_dataset", methods=["POST"])
def api_capture_to_dataset():
    data   = request.json or {}
    name   = (data.get("person_name") or "").strip()
    frames = data.get("frames", [])
    if not name:
        return jsonify({"ok": False, "error": "person_name is required"}), 400
    if not frames:
        return jsonify({"ok": False, "error": "No frames provided"}), 400

    person_dir = DATASET / name
    person_dir.mkdir(parents=True, exist_ok=True)
    saved, skipped = 0, 0

    for f in frames:
        bgr = decode_base64_frame(f)
        if bgr is None:
            skipped += 1; continue
        if not detect_faces(bgr):
            skipped += 1; continue
        fname = f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(person_dir / fname), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1

    return jsonify({
        "ok":      saved > 0,
        "saved":   saved,
        "skipped": skipped,
        "person":  name,
        "message": f"Captured {saved}/{len(frames)} valid frame(s) for '{name}'"
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    return jsonify(train_model())


@app.route("/api/delete_person", methods=["POST"])
def api_delete_person():
    name = (request.json or {}).get("name", "").strip()
    person_dir = DATASET / name
    if not person_dir.exists():
        return jsonify({"ok": False, "error": "Person not found"}), 404
    shutil.rmtree(person_dir)
    return jsonify({"ok": True,
                    "message": f"'{name}' removed. Re-train to update model."})


@app.route("/api/get_logs")
def api_get_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, date, time, confidence FROM logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify({"logs": rows})


@app.route("/api/clear_logs", methods=["POST"])
def api_clear_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM logs")
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "message": "All attendance logs cleared."})


# ── Backwards-compat shims ──
@app.route("/identify", methods=["POST"])
def legacy_identify():
    return api_recognize()


@app.route("/get_logs")
def legacy_get_logs():
    return api_get_logs()


# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_model()
    print("\n" + "=" * 64)
    print("  Site 2 — Attendance System (PATCHED)")
    print(f"  Model     : {'loaded ✓' if model_loaded else 'NOT TRAINED — visit /api/train'}")
    print(f"  Algorithm : MediaPipe (short range) → Align → PCA → LDA → SVM")
    print(f"  Fixes     : grayscale support · close-range detection · ")
    print(f"              honest train/test split · tight-crop fallback · box overlay")
    print(f"  URL       : http://127.0.0.1:5000")
    print("=" * 64 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
