
# 🧠 Image Segmentation using U-Net (Oxford-IIIT Pet)

A **pixel-wise semantic image segmentation** project using a custom **U-Net architecture** implemented from scratch in PyTorch.
The model learns to separate foreground objects (pets) from background at the **pixel level**, going beyond standard image classification.

This project is designed to be **CPU-friendly**, reproducible, and aligned with **real-world computer vision workflows**, especially those used in **medical and scientific imaging**.

---

## 🔍 Problem Statement

Given an input image, predict a **binary segmentation mask** where:

* `1` → object of interest (pet)
* `0` → background

Unlike classification, segmentation requires **dense prediction**, meaning the model must correctly classify **every pixel** in the image.

---

## 📊 Dataset

**Oxford-IIIT Pet Dataset**

* Images of cats and dogs
* Pixel-level segmentation masks (trimaps)
* Original masks contain:

  * Background
  * Object body
  * Boundary pixels

### Preprocessing applied

* Masks converted to **binary segmentation**

  * Object + boundary → foreground
  * Background → background
* Images resized to **128×128**
* Nearest-neighbor interpolation used for masks
* Normalized tensors

This setup mirrors preprocessing pipelines used in **medical image segmentation**.

---

## 🏗️ Model Architecture

### U-Net (from scratch)

* Encoder–decoder structure
* Skip connections to preserve spatial details
* Fully convolutional (no dense layers)
* Output shape matches input spatial resolution

**Input:**
`[B, 3, 128, 128]`

**Output:**
`[B, 1, 128, 128]` (raw logits)

---

## 📉 Loss Function & Metrics

### Loss

A combination of:

* **Binary Cross-Entropy with Logits**
* **Dice Loss**

This balances:

* Pixel-wise accuracy
* Region overlap quality (important for segmentation)

### Evaluation Metric

* **Dice Coefficient**

  * Measures overlap between predicted mask and ground truth
  * Commonly used in medical imaging tasks

---

## ⚙️ Training Details

* Framework: **PyTorch**
* Device: **CPU**
* Optimizer: Adam
* Learning rate: `1e-3`
* Batch size: `8`
* Epochs: `5`
* Train/Validation split: `80 / 20`

Model checkpoint is saved after training.

---

## 📁 Project Structure

```
segmentation-unet/
│
├── data/                # Dataset (auto-downloaded)
├── runs/                # Saved models
├── notebooks/           # (optional) experiments
├── src/
│   ├── dataset.py       # Dataset + preprocessing
│   ├── unet.py          # U-Net implementation
│   ├── train.py         # Training loop
│   ├── sanity_check.py  # Environment test
│   ├── check_unet.py    # Model shape check
│   └── check_binary_mask.py
│
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Create environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib tqdm opencv-python scikit-learn
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Output

* Training & validation Dice scores printed per epoch
* Model saved to:

```
runs/unet_pet_baseline.pth
```

---

## 📈 Results

* Model successfully learns to segment pets from background
* Clean binary masks predicted after a few epochs
* Demonstrates correct learning of object boundaries and shape

Qualitative evaluation via mask overlays confirms meaningful segmentation.

---

## 🔮 Future Improvements

* Attention U-Net
* Transfer-learning encoder (MobileNet / ResNet)
* Higher resolution inputs
* Multiclass segmentation
* Application to **medical imaging datasets** (skin lesions, organs)
* Quantitative IoU analysis
* Explainability for segmentation outputs

---

## 🎯 Why This Project Matters

This project demonstrates:

* Understanding of **dense prediction problems**
* Proper handling of segmentation datasets
* Custom architecture implementation
* Practical evaluation beyond accuracy
* Research-oriented computer vision skills

It serves as a strong **bridge between coursework CV projects and research-grade segmentation work**.

