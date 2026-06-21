
CPCS-331 Major Project — King Abdulaziz University

A complete face-recognition system with a Flask web interface, real-time
webcam recognition, automatic attendance logging, and a documented
classical machine-learning pipeline.

> **Note on this public version:** the original project includes a
> `dataset/` folder of training photos, a trained `model.pkl`, and an
> `attendance.db` log. These are **excluded here** because the dataset
> contains identifiable photos of the project team, and the trained
> model/database are derived directly from those photos. See
> [Privacy Notes](#-privacy-notes) below for details. The code is
> otherwise unchanged.

---

## 📦 What's In This Project

```
ProjectFolder/
├── README.md                   ← you are here
├── HOW_TO_RUN.txt              ← detailed setup + run instructions (READ THIS)
├── app.py                      ← main Flask web application
├── index.html                  ← website frontend (4 tabs)
│
├── run.py                      ← optional one-click launcher
├── eda.py                      ← exploratory data analysis script
├── evaluate.py                 ← model evaluation script
├── hypertune.py                ← hyperparameter optimization script
├── extra_visuals.py            ← additional report figures
│
├── charts/                     ← exported plots (confusion matrix, eigenfaces, etc.)
│
└── Presentation.html           ← defense presentation
```

Not included in this repo (see [Privacy Notes](#-privacy-notes)):
`dataset/` (training photos), `model.pkl` (trained model), `attendance.db`
(attendance log).

---

## 🚀 Quick Start

For complete step-by-step instructions, **see `HOW_TO_RUN.txt`**.

The short version:

```bash
cd path\to\this\folder
python -m venv venv               # first time only
venv\Scripts\activate
pip install flask opencv-python opencv-contrib-python numpy scikit-learn Pillow mediapipe pandas seaborn matplotlib    # first time only
python app.py
```

Then open **http://127.0.0.1:5000** in any modern browser.

Since the trained `model.pkl` isn't included, the live recognition demo
won't have a model to load out of the box. To try it end-to-end, add your
own photos under `dataset/<your_name>/` and run `hypertune.py` to train
and save a new `model.pkl`.

---

## 🧠 The Pipeline

```
Webcam frame
   │
   ▼
MediaPipe Face Detection    →  bounding box + 6 facial keypoints
   │
   ▼
Affine Alignment            →  eyes and nose mapped to fixed positions
   │
   ▼
100×100 grayscale crop      →  flatten to 10,000-dim vector
   │
   ▼
PCA (50 components)         →  dimensionality reduction (Eigenfaces)
   │
   ▼
LDA (3 components)          →  maximize class separation (Fisherfaces)
   │
   ▼
SVM (RBF kernel)            →  classify with probability
   │
   ▼
Threshold check             →  if prob < threshold → "Unknown"
   │
   ▼
Result + attendance log
```

---

## 📊 Results

| Metric                         | Value                 
|------------------------------------------
| Classes                        | 6 (5 registered identities + Unknown) |
| Total samples                  | 407 |
| Train / Test split             | 325 / 82 |
| PCA components                 | 50 |
| LDA components                 | 3 |
| PCA variance explained         | ~93% |
| Test accuracy                  | 98.78%         |
| Best hyperparams (grid search) | PCA=50, C=10, gamma=scale |

**PCA captures 93.6% of variance in just 50 of 10,000 pixel dimensions:**

<img width="1305" height="707" alt="pca_variance_curve" src="https://github.com/user-attachments/assets/9945075f-cc8f-49e2-b336-705cdbd4bb10" />


**The first 16 eigenfaces — the principal directions of facial variation
the model learns:**

<img width="1290" height="1350" alt="eigenfaces" src="https://github.com/user-attachments/assets/cfb85011-1467-44c1-b07e-b8e4a6c5af36" />


**LDA projection — each person forms a distinct, separable cluster:**

<img width="1305" height="947" alt="lda_clusters" src="https://github.com/user-attachments/assets/a368f21e-bae0-4aef-a8e7-6869c2c92ff9" />


**Confusion matrix on the held-out test set (98.8% accuracy):**

<img width="1142" height="957" alt="confusion_matrix" src="https://github.com/user-attachments/assets/2c8d5674-6297-43f6-9bfc-960a8392cd7c" />


---

## 🌐 The Web Interface

The site has four tabs:

1. **🎥 Live Camera** — webcam recognition with real-time green/red boxes
   overlaid on the live video; logs attendance automatically.
2. **📷 Upload Photo** — identify faces in an uploaded image; does NOT
   log attendance.
3. **👤 Manage People** — capture new people from webcam, upload photos,
   delete entries, retrain the model.
4. **📊 Attendance Log** — view, refresh, or clear attendance records.

---

## ⚠️ Known Limitations

The system uses classical machine learning, which has well-documented
weaknesses:

- **Lighting changes** — the most pronounced weakness
- **Partial occlusion** — covering the face still often gives a result
- **Accessory changes** — glasses, hats, sunglasses degrade recognition
- **Appearance changes** — significant beard / haircut / hair color
- **Strong pose** — yaw rotation greater than ~45°

A deep-learning extension (FaceNet / ArcFace) would address most of
these and is the natural next step for this project.

---

## 🛠️ Tech Stack

- **Backend:** Python · Flask · SQLite
- **Computer Vision:** MediaPipe · OpenCV
- **Machine Learning:** scikit-learn (PCA, LDA, SVM)
- **Frontend:** HTML5 · JavaScript · HTML5 Canvas + getUserMedia API

---

## 🔒 Privacy Notes

This public version of the project intentionally excludes three items
present in the original submission:

- **`dataset/`** — training photos of the project team. Excluded
  because the images are identifiable photos of real people who didn't
  consent to public posting.
- **`model.pkl`** — the trained PCA/LDA/SVM model. Excluded because
  PCA/LDA models like this store linear projections (mean face,
  eigenfaces, per-class means) built directly from the training photos,
  which can be used to reconstruct a recognizable approximation of each
  person's face even without the original images.
- **`attendance.db`** — the attendance log, which ties real names to
  timestamps. Excluded for the same reason as the dataset.

The charts kept in this repo (eigenfaces, mean face, confusion matrix,
PCA/LDA plots, etc.) are aggregate statistics and do not expose any
individual's photo.

---

## 📚 Key References

1. Belhumeur, P. N., Hespanha, J. P., & Kriegman, D. J. (1997).
   *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear
   Projection.* IEEE Transactions on Pattern Analysis and Machine
   Intelligence, 19(7), 711–720.
2. Mitchell, T. M. (1997). *Machine Learning.* McGraw-Hill.
3. Russell, S. J., & Norvig, P. (2020). *Artificial Intelligence: A
   Modern Approach* (4th ed.). Pearson.
4. Lugaresi, C. et al. (2019). *MediaPipe: A Framework for Building
   Perception Pipelines.* Google Research.

---

## 📞 More Details

- **For setup and running:** see `HOW_TO_RUN.txt`
- **For the defense presentation:** see `Presentation.html`
