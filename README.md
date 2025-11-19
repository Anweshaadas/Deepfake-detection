# 🤖 Deepfake Detection using EfficientNet and Transfer Learning

## 📖 Overview

This project implements a robust deep learning solution for classifying video frames as either **REAL** (authentic) or **FAKE** (deepfake/manipulated). The core algorithm utilizes **Transfer Learning** on the **EfficientNetB0** architecture, leveraging its powerful feature extraction capabilities pre-trained on ImageNet.

The project is structured as a simple, single-file pipeline (`train_model.py`) focused on training and detailed evaluation.

##  Key Features

* **Transfer Learning:** Uses a frozen **EfficientNetB0** backbone for highly accurate feature extraction, minimizing training time.
* **Binary Classification:** A simple classification head (Global Average Pooling, Dropout, Dense layer) determines the final REAL/FAKE verdict.
* **Frame-Level Analysis:** Designed to process pre-extracted, cropped face images for maximum focus on manipulation artifacts.
* **Clear Evaluation:** Reports detailed metrics including Precision, Recall, F1-Score, and a Confusion Matrix.

---

##  Setup and Requirements

### 1. Prerequisites

* **Python 3.8+**
* **Git** (for cloning)
* **Sufficient Disk Space:** Deep learning libraries and data require several GBs of free space.

### 2. Environment Setup

It is strongly recommended to use a **Python Virtual Environment** (`.venv`).

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/Deepfake-Detection-Project.git](https://github.com/your-username/Deepfake-Detection-Project.git)
cd Deepfake-Detection-Project

# 2. Create and activate the virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\activate
