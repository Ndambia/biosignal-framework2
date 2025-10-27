# BioSignal Framework

**A modular open-source framework for acquisition, preprocessing, feature extraction, and classification of biosignals (EMG, ECG, EOG).**

---

## 📘 Overview

The **BioSignal Framework** is a unified, extensible research toolkit designed for biosignal processing and machine learning applications.
It provides a consistent API for handling **electromyography (EMG)**, **electrocardiography (ECG)**, and **electrooculography (EOG)** signals—covering the entire pipeline from signal acquisition to feature extraction and classification.

This framework is intended for:

- Biomedical signal processing research
- Embedded ML biosignal prototyping
- Wearable and neurophysiology applications
- Machine learning model development and benchmarking

---

## 🧩 Architecture

The framework follows a **layered modular architecture**, allowing flexibility and reusability.

```
biosignal_framework/
│
├── acquisition/          # Signal loading & acquisition interfaces
│   ├── emg_acquisition.py
│   ├── ecg_acquisition.py
│   ├── eog_acquisition.py
│
├── preprocessing/        # Signal denoising, normalization, segmentation
│   ├── filters.py
│   ├── segmentation.py
│
├── features/             # Feature extraction across domains
│   ├── time_domain.py
│   ├── frequency_domain.py
│   ├── nonlinear_features.py
│
├── models/               # ML model training and inference
│   ├── classifier.py
│   ├── pipeline.py
│
├── utils/                # Helper utilities (I/O, visualization, etc.)
│   ├── io_utils.py
│   ├── visualization.py
│
├── tests/                # Unit tests for all components (pytest)
│
├── notebooks/
│   └── demo_emg_pipeline.ipynb  # Example: EMG acquisition → preprocessing → features → ML
│
└── README.md
```

---

## ⚙️ Installation

### Requirements

Python ≥ 3.9
Required libraries:

```bash
numpy scipy matplotlib scikit-learn pywt pandas
```

### Setup

```bash
git clone https://github.com/<your-username>/biosignal-framework.git
cd biosignal-framework
pip install -r requirements.txt
```

---

## 🧠 Core Concepts

### 1. Acquisition

Supports offline and real-time acquisition via:

- File-based loaders (CSV, EDF, MAT)
- Serial or BLE streaming (for embedded sensors)
- Synthetic generators for testing

### 2. Preprocessing

Implements:

- Bandpass and notch filtering
- Baseline correction
- Segmentation
- Wavelet denoising

### 3. Feature Extraction

Feature sets include:

- **Time-domain:** Mean, RMS, MAV, Zero-Crossing, IQR
- **Frequency-domain:** PSD, spectral centroid, median frequency
- **Nonlinear:** Entropy, fractal dimension

### 4. Modeling

Supports:

- Classical ML models (SVM, RandomForest, LinearRegression)
- Deep learning integration (PyTorch, TensorFlow ready)
- Pipeline orchestration for training and validation

---

## 📊 Example Workflow

The included Jupyter Notebook (`notebooks/demo_emg_pipeline.ipynb`) demonstrates:

```python
from biosignal_framework.acquisition.emg_acquisition import EMGAcquisition
from biosignal_framework.preprocessing.filters import bandpass_filter
from biosignal_framework.features.time_domain import extract_time_features
from biosignal_framework.models.pipeline import BioSignalPipeline

# 1. Acquire
signal = EMGAcquisition().load('data/sample_emg.csv')

# 2. Preprocess
filtered = bandpass_filter(signal, 20, 450, fs=1000)

# 3. Extract Features
features = extract_time_features(filtered)

# 4. Model Training
pipeline = BioSignalPipeline(model='svm')
pipeline.train(features, labels)
```

---

## 🧪 Testing

Unit tests are located in the `tests/` directory.
To run all tests:

```bash
pytest -v
```

---

## 🔄 Continuous Integration

This repository includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:

- Lints the codebase with **flake8**
- Runs **pytest** for all modules

---

## 📈 Future Extensions

- Real-time streaming interface via BLE & UART
- Deep feature learning (CNN, LSTM)
- Cross-signal fusion (EMG + ECG + EOG)
- Embedded deployment (TensorFlow Lite / Edge Impulse)

---

## 🧑‍🔬 Citation

If you use this framework in your research, please cite:

> **Mwangi, B. (2025). BioSignal Framework: A Modular Platform for EMG, ECG, and EOG Processing.**
> GitHub Repository: [https://github.com/Ndambia/biosignal-framework](https://github.com//Ndambia//biosignal-framework)

---

## 📜 License

MIT License © 2025 Brian Mwangi
Open for academic and industrial collaboration.

---
