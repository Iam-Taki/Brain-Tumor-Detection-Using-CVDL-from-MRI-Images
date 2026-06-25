# Brain-Tumor-Detection-Using-CVDL-from-MRI-Images


**Author:** Abdullah Al Noman Taki (VR528988)
**Course:** Computer Vision & Deep Learning — University of Verona
**Program:** MSc in Artificial Intelligence, A.Y. 2024-25

---

## 📋 Overview

This project implements and compares **five deep learning architectures** for multi-class brain tumor classification using MRI images. The goal is to automatically classify brain scans into four diagnostic categories, supporting radiologists in early and accurate tumor detection.

## 🧠 Dataset

- **Samples:** 7,023 brain MRI images
- **Classes:**
  | Class | Description |
  |---|---|
  | Glioma | Tumor arising from glial cells |
  | Meningioma | Tumor arising from the meninges |
  | Pituitary | Tumor of the pituitary gland |
  | No Tumor | Healthy brain scan |

- **Split:** 5,712 training images / 1,311 test images
- **Preprocessing:** Resized to 128×128, normalized to [0,1], augmented with random brightness and contrast adjustments

## 🏗️ Models Implemented

| Model | Type | Test Accuracy | Macro F1 | Macro AUC |
|---|---|---|---|---|
| **VGG16** 🏆 | Transfer learning | **96.72%** | **0.9653** | **0.9964** |
| MobileNetV2 | Transfer learning (lightweight) | 93.97% | 0.9363 | 0.9941 |
| U-Net Encoder | Trained from scratch | 90.85% | 0.9018 | 0.9847 |
| ResNet50 | Transfer learning | 73.07% | 0.6726 | 0.9475 |
| CNN + LSTM | Trained from scratch (hybrid) | 70.63% | 0.6656 | 0.9142 |

🏆 **Best Model: VGG16** — effective transfer learning with fine-tuned final layers achieved the strongest generalization on this dataset.

## 🔬 Ablation Study (VGG16)

To examine hyperparameter sensitivity, three controlled variants of the best-performing model (VGG16) were trained:

| Variant | Change | Test Accuracy |
|---|---|---|
| Baseline | Learning rate = 0.0001 | **96.72%** |
| Variant A | Learning rate = 0.001 | 30.89% (training collapsed) |
| Variant B | Dropout = 0.5 (vs. 0.3) | 96.19% |
| Variant C | Last 6 layers fine-tuned (vs. 3) | 93.14% |

**Key finding:** Learning rate is by far the most sensitive hyperparameter for transfer learning — a 10× increase destabilized the pretrained ImageNet weights and caused training to collapse.

## 🔍 Explainability — Grad-CAM

Since Captum (PyTorch-only) is incompatible with our TensorFlow/Keras models, **Grad-CAM** was used as the explainability technique. Activation maps were generated for correctly classified MRI images to verify that model predictions are grounded in clinically relevant tumor regions rather than background noise.

## ⚙️ Preprocessing Pipeline

1. Load raw MRI images and labels from class-organized folders
2. Split into training (85%) / validation (15%) / test sets (stratified)
3. Resize all images to 128×128×3
4. Apply real-time augmentation (brightness ±20%, contrast ±20%) during training
5. Normalize pixel values to [0, 1]
6. Encode class labels to integers for `sparse_categorical_crossentropy`

## 🔧 Hyperparameters (Unified Across All Models)

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 20 |
| Epochs | 10 |
| Loss | Sparse Categorical Crossentropy |
| Fine-tuned Layers | Last 3 layers of pretrained backbone |

## 📊 Evaluation Metrics

Accuracy, Precision, Recall, F1-Score (macro & weighted), Confusion Matrix, and ROC-AUC (one-vs-rest).

## 📁 Repository Structure

```
├── Building_a_Brain_Tumor_Detection_Using_Deep_Learning.ipynb   # Full pipeline: preprocessing, 5 models, ablation, Grad-CAM
├── report/
│   └── Brain_Tumor_Detection_Report.pdf
├── figures/
│   ├── architecture_diagrams/        # VGG16, ResNet50, MobileNetV2, U-Net, CNN+LSTM
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── final_comparison_all_models.png
│   ├── ablation_study_vgg16.png
│   └── gradcam_examples/             # One per class
└── README.md
```

## 🚀 How to Run

This notebook was trained locally using an **NVIDIA RTX 4060 GPU** (TensorFlow 2.10, CUDA 11.2, cuDNN 8.1 via Miniconda on Windows).

1. Set up a conda environment: `conda create -n cvdl python=3.10`
2. Install CUDA/cuDNN: `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1`
3. Install TensorFlow: `pip install tensorflow==2.10 numpy==1.23.5`
4. Update the `train_dir` / `test_dir` paths to point to your local MRI Images folder
5. Run all cells sequentially in Jupyter Lab/Notebook

> Originally prototyped on Google Colab; later migrated to a local GPU setup for significantly faster training (VGG16: ~3.5 hours on Colab CPU → ~2 minutes on RTX 4060).

## 📈 Key Findings

- **Transfer learning dominates**: VGG16 and MobileNetV2 (both ImageNet-pretrained) substantially outperformed models trained from scratch (U-Net, CNN+LSTM) and the underperforming ResNet50
- **Learning rate is the most critical hyperparameter** for fine-tuning pretrained networks — even a moderate increase can destroy pretrained representations
- **Glioma was the most difficult class** for ResNet50 and CNN+LSTM, with recall as low as 0.20–0.29, while VGG16 maintained strong performance across all four classes
- **Grad-CAM confirmed clinical relevance**: the best model's attention was concentrated on tumor regions rather than irrelevant background structures, particularly for glioma cases

## 📚 References

1. Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
3. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
4. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
7. Deng, J., et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. *CVPR*.
8. Cheng, J. (2017). Brain Tumor Dataset. *Figshare*.

---

*This project was developed as part of the Machine Learning & Deep Learning and Computer Vision & Deep Learning courses at the University of Verona.*
