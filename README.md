# FMML Capstone Project – Semantic Segmentation for Indian Driving Dataset (IDD)

Welcome to the repository for my FMML (Foundations of Modern Machine Learning) Capstone Project. This project tackles **semantic segmentation** using deep learning techniques on the **Indian Driving Dataset (IDD)**. The focus is on **pixel-wise classification** of road scenes captured across various Indian cities into 26 level-3 classes, aiding perception systems for autonomous vehicles.

## 🚗 Problem Statement

Autonomous driving in India presents unique challenges due to unstructured roads, inconsistent infrastructure, and high visual variability. To address this, we perform **semantic segmentation** - a computer vision task that classifies each pixel in an image into predefined categories relevant to road environments (e.g., roads, vehicles, pedestrians).

## 🎯 Objectives

- Develop a deep learning model to perform semantic segmentation on IDD data.
- Achieve high accuracy using the **Mean Intersection over Union (mIoU)** metric.
- Optimize training with best practices such as data augmentation, checkpointing, and validation monitoring.

## 🧪 Dataset: Indian Driving Dataset (IDD)

- **Size:** 20,000+ images (IDD-20K Part I & II)
- **Annotations:** Level-3 labels for 26 semantic classes + 1 miscellaneous class
- **Resolution:** Resized to 256 x 256 during training; original size 1280 x 720
- **Splitting:** 85% training / 15% validation

## 🧰 Tools & Frameworks

- **Framework:** PyTorch & TorchVision
- **Model:** DeepLabV3+ with MobileNetV2 backbone
- **Training Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss with masking
- **Evaluation Metric:** Mean IoU (mIoU)

## 📊 Model Training & Evaluation

The notebook (FMML-Capstone-Project.ipynb) performs the following steps:

### 🔄 Data Preparation

- Resizing all images/masks to 256x256
- Applied augmentations: random flip, rotation, color jitter
- Split dataset into training and validation sets

### 🏗️ Model Architecture

- DeepLabV3+ encoder-decoder with MobileNetV2 as backbone
- Lightweight yet accurate segmentation model

### ⚙️ Training Configuration

- Batch size: 32
- Epochs: 10
- Learning Rate: 1e-3
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

### 📈 Results

- Best validation mIoU: 0.8520 (at Epoch 8)
- Training loss and accuracy monitored per epoch
- Saved model checkpoints using `torch.save`

### 🧪 Evaluation

- Computed per-class IoU
- Visualized predictions for qualitative assessment
- Identified confusion in similar categories (e.g., pedestrian vs rider)

## 🔍 Challenges Faced

- Slight overfitting after epoch 8
- Class imbalance in rare categories
- Tradeoff between training speed and input resolution

## 🔮 Future Work

- Fine-tune on higher resolution (512 x 512 or 1280 x 720)
- Test transformer-based models like SegFormer, Mask2Former
- Perform domain adaptation with Cityscapes and BDD100K

## 🧪 Installation & Setup

Clone this repository:

```bash
git clone https://github.com/Sadiya-025/FMML-Capstone-Project.git
cd FMML-Capstone-Project
```

Create virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the notebook:

```bash
jupyter notebook FMML-Capstone-Project.ipynb
```

Prepare dataset:

- Download IDD Part I & II
- Place and extract them inside the `domain_adaptation/` directory
- Run the official label generation script (`createLabels.py`)

## 🏆 Performance Summary

| Metric       | Value  |
| ------------ | ------ |
| Best mIoU    | 0.8520 |
| Epoch        | 8      |
| Val Accuracy | ~95.6% |

## 📬 Contact

- **Author:** Sadiya Maheen Siddiqui
- **Portfolio:** [sadiya-maheen-siddiqui.vercel.app](https://sadiya-maheen-siddiqui.vercel.app)
