# Fruit Quality Grading System Using Image Processing and Machine Learning

An end-to-end computer vision and machine learning web application that automates fruit quality inspection and freshness grading. The system combines deep salient object segmentation (**U²-Net** via `rembg`), colour and texture feature extraction (**HSV colour spaces** and **Gray-Level Co-occurrence Matrix (GLCM)**), and an ensemble **Random Forest Classifier** to accurately grade fruits as either **Fresh** or **Rotten**.

Developed for academic demonstration with a modern, responsive agricultural-technology dashboard built using **Streamlit**.

---

## 📋 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Key Features](#-key-features)
- [Dataset Details](#-dataset-details)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Experimental Evaluation](#-experimental-evaluation)
- [Project Directory Structure](#-project-directory-structure)
- [Installation Guide (Windows)](#-installation-guide-windows)
- [Running the Application](#-running-the-application)
- [System Limitations](#-system-limitations)

---

## 🌟 Overview & Architecture

Inspecting fruit freshness manually in wholesale and retail agriculture is labor-intensive, subjective, and prone to error. This project delivers an automated, non-destructive optical grading system that processes digital photographs of fruits through an automated multi-stage pipeline:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Fruit Image    │ ───►  │  U²-Net Deep    │ ───►  │  Alpha-Masked   │
│  (JPG / PNG)    │       │  Segmentation   │       │  Fruit Object   │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                                             │
                                                             ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Final Prediction│ ◄───  │  Random Forest  │ ◄───  │ 7 Numerical     │
│ (Fresh/Rotten)  │       │  (300 Trees)    │       │ Features        │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

1. **Input Acquisition:** Receives high-resolution fruit photos from standard cameras or smartphones.
2. **U²-Net Segmentation:** Strips background clutter, tables, plates, and ambient cast shadows using a deep salient object detection neural network.
3. **Feature Extraction:** Measures 3 HSV colour metrics and 4 horizontal adjacent GLCM texture metrics.
4. **Ensemble Classification:** Evaluates features using a trained 300-tree Random Forest classifier.
5. **Interactive Dashboard:** Visualizes results, model probabilities, segmentation masks, feature statistics, and historical audit logs.

---

## ✨ Key Features

- **Multi-Class Multi-Fruit Support:** Identifies 8 different fruit species and predicts binary quality (Fresh vs. Rotten) across 16 total output classes.
- **Robust Background Removal:** Employs U²-Net neural network with automated fallback alpha thresholds (128 → 32 → 0) to ensure foreground fruit surface isolation.
- **Scientifically Formulated Features:** Extracts exact colour distribution metrics (Hue, Saturation, Value) and GLCM spatial texture metrics (Contrast, Homogeneity, Energy, Correlation).
- **Interactive Multi-Page Streamlit Dashboard:**
  - **🏠 Home / Dashboard:** Overview, workflow explanation, and benchmark summaries.
  - **🔍 Fruit Quality Analysis:** Real-time upload, segmentation, and classification with model probability scores and candidate class distribution.
  - **🖼️ Image Segmentation:** Side-by-side comparison of original image, U²-Net alpha mask, and isolated fruit.
  - **📊 Feature Analysis:** Interactive tables and charts explaining the numerical contribution of colour and texture.
  - **ℹ️ Model Information:** Full architectural hyperparameters, classification report, and confusion matrix.
  - **📜 Prediction History:** Persistent local SQLite database (`prediction_history.db`) with filtering, statistics, and CSV export.

---

## 🍓 Dataset Details

The system is trained and evaluated on 8 common commercial fruit varieties:

| Fruit Category | Quality Classes | Number of Target Classes |
| :--- | :--- | :--- |
| **Apple** | Fresh, Rotten | 2 |
| **Banana** | Fresh, Rotten | 2 |
| **Grape** | Fresh, Rotten | 2 |
| **Guava** | Fresh, Rotten | 2 |
| **Jujube** | Fresh, Rotten | 2 |
| **Orange** | Fresh, Rotten | 2 |
| **Pomegranate** | Fresh, Rotten | 2 |
| **Strawberry** | Fresh, Rotten | 2 |
| **Total** | **2 Quality States** | **16 Prediction Classes** |

The training dataset comprises 3,200 segmented image records (200 images per target class), split 80% for training (2,560 images) and 20% for testing (640 images) using stratified sampling.

---

## 🔬 Feature Engineering

The model takes **exactly seven input features** in the following designated order:

1. **`mean_hue`** *(HSV)*: Dominant skin pigmentation. Differentiates fruit species (e.g., green guava vs. red strawberry) and ripening/oxidation browning.
2. **`mean_saturation`** *(HSV)*: Color vividness. Fresh produce exhibits vibrant, saturated skin, whereas aging fruit appears pale or greyish.
3. **`mean_brightness`** *(HSV)*: Surface luminance. Fungal growth, rot spots, and necrosis cause localized and global darkening.
4. **`contrast`** *(GLCM)*: Gray-level intensity variations between adjacent pixels. Bruised, moldy, and pitted surfaces yield higher contrast.
5. **`homogeneity`** *(GLCM)*: Surface uniformity. Smooth, undamaged fruit skins display high homogeneity.
6. **`energy`** *(GLCM)*: Angular Second Moment measuring textural regularity.
7. **`correlation`** *(GLCM)*: Spatial linear dependence of gray levels across the fruit surface.

All texture calculations use 8-bin grayscale quantization (`gray // 32`) on horizontally adjacent pixel pairs (`mask[:, :-1] & mask[:, 1:]`).

---

## 📊 Experimental Evaluation

The serialized Random Forest model (`models/best_fruit_quality_model.pkl`) was trained with the following hyperparameter settings:
- **Number of Estimators:** `300`
- **Class Weight:** `balanced_subsample`
- **Random State:** `42`
- **Execution Mode:** `n_jobs=-1` (multithreaded CPU)

### Recorded Offline Benchmark Results (640-sample stratified test split):
- **Overall Accuracy:** `80.47%`
- **Macro Average F1-Score:** `80.24%`
- **Weighted Average F1-Score:** `80.24%`

> **Note on Evaluation:** These figures reflect controlled offline test set benchmarks. Real-world performance will naturally vary depending on camera resolution, lighting uniformity, and foreign occlusions.

---

## 📁 Project Directory Structure

```
Fruit-Quality-Grading/
│
├── app.py                             # Main Streamlit web application entry point
├── requirements.txt                   # Application dependencies
├── README.md                          # Complete project documentation
├── fruit_features_segmented.csv       # Extracted features dataset (3,200 records)
├── .streamlit/
│   └── config.toml                    # Streamlit theme and upload configuration
│
├── app_modules/                       # Reusable application modules
│   ├── __init__.py
│   ├── model_loader.py                # Cached model loader and class label mapping
│   ├── image_processing.py            # U²-Net background removal and mask extraction
│   ├── feature_extraction.py          # Exact HSV & GLCM feature calculations
│   ├── prediction.py                  # Random Forest inference & probability scoring
│   └── database.py                    # SQLite operations & query functions
│
├── models/
│   ├── best_fruit_quality_model.pkl   # Preserved trained Random Forest model
│   └── evaluate_best_model.py         # Evaluation and validation script
│
├── preprocessing/
│   ├── extract_features.py            # Original dataset feature extraction pipeline
│   └── automatic_segmentation.py      # GrabCut segmentation experiment
│
├── Original Image/                    # Original fruit dataset folders (16 categories)
├── segmented_classification_report.csv# Test classification metrics table
├── segmented_feature_importance.csv   # Feature importance rankings
└── segmented_confusion_matrix.png     # Saved evaluation confusion matrix plot
```

The original image dataset, generated evaluation outputs, and local SQLite
history database are not required for deployment. Dataset samples will not be
available in the app unless provided separately; image uploads work without
them.

---

## 💻 Installation Guide (Windows)

### Prerequisites
- Python 3.12 installed on Windows.
- Standard Command Prompt or PowerShell.

### Step 1: Open Terminal in Project Folder
Navigate to the project root:
```powershell
cd C:\Users\nihar\Downloads\Fruit-Quality-Grading
```

### Step 2: Install Required Dependencies
Install all required packages from `requirements.txt`:
```powershell
py -m pip install -r requirements.txt
```
*(Alternatively, if `py` is not in PATH: `python -m pip install -r requirements.txt`)*

> **Internet Connection Note:** On the first execution, `rembg` may download the U²-Net pre-trained ONNX model weights (`u2net.onnx`, ~176 MB) into your user profile directory (`~/.u2net/`). Subsequent runs operate completely offline.

---

## 🚀 Running the Application

Launch the Streamlit web server:

```powershell
py -m streamlit run app.py
```
*(Or `python -m streamlit run app.py`)*

The application will launch automatically in your default browser at:
`http://localhost:8501`

## ☁️ Deploying from GitHub with Streamlit Community Cloud

1. Push the application files to GitHub, including `app.py`, `app_modules/`,
   `models/best_fruit_quality_model.pkl`, `requirements.txt`, and
   `.streamlit/config.toml`. The hero image under `assets/` is optional.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app
   from the repository and select the branch and `app.py` as the main file.
3. Select Python **3.12** in the app's advanced settings, then deploy.

The pinned dependencies were checked with the included model on Python 3.12.
The model file is about 66 MiB and must be present in the repository at the
path above. The first analysis may download U²-Net model weights (~176 MB), so
the app needs outbound network access and enough memory for CPU inference.
Prediction history is stored in a local SQLite file and is ephemeral on
Streamlit Community Cloud; it may be lost when the app restarts or redeploys.

The `Original Image/` dataset is excluded by `.gitignore` to avoid uploading
thousands of sample photos. The app supports uploaded images without it.

---

## ⚠️ System Limitations & Academic Considerations

1. **Surface Inspection Only:** The optical system inspects surface skin. Internal rot, seed degradation, or heart rot that does not express surface symptoms cannot be detected via standard RGB imaging.
2. **Lighting Sensitivity:** Extreme specular highlights, reflections, or very dim environments can shift calculated Hue and Brightness values.
3. **Occlusions & Angles:** If a rotten patch is positioned on the underside hidden from the camera lens, the model will classify based solely on the visible surface.
4. **Known Species Scope:** The model is specialized for the 8 trained fruit species. Unseen exotic fruit types will be matched to the closest statistical distribution among the 16 target classes.
