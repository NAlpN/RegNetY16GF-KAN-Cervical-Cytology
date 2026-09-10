# RegNetY-16GF + Kolmogorov-Arnold Networks (KANs) for Cervical Cytology Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: IEEE](https://img.shields.io/badge/IEEE-ICECER'26-00629B.svg)](https://www.ieee.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Official PyTorch implementation of the **Hybrid RegNetY-16GF + Kolmogorov-Arnold Network (KAN)** pipeline for automated multi-class classification and explainable artificial intelligence (XAI) in **Liquid-Based Cervical Cytology (CCID)**.

---

## 📌 Architecture Overview

<p align="center">
  <img src="assets/proposed_regnet_kan_architecture.png" alt="Proposed RegNetY-16GF + KAN Architecture" width="100%"/>
</p>
<p align="center">
  <em>Figure 1: End-to-end architecture pipeline featuring cytological data augmentation, RegNetY-16GF backbone with Squeeze-and-Excitation (SE) bottleneck blocks, global average pooling (GAP), linear projection with LayerNorm and GELU, and a two-layer Kolmogorov–Arnold Network (KAN) classifier head.</em>
</p>

---

## 🔬 Abstract & Key Contributions

Accurate cervical cytology screening is vital for early diagnosis and treatment of cervical cancer. Standard Deep Convolutional and Vision Transformer architectures frequently struggle with intra-class cytomorphological variations, stain fluctuations, and the risk of overfitting in delicate cellular imaging.

To address these challenges, we introduce a hybrid architecture coupling **RegNetY-16GF** with **Kolmogorov-Arnold Networks (KANs)**:

1. **RegNetY-16GF Backbone**: Leverages regular design spaces optimized with **Squeeze-and-Excitation (SE)** residual blocks to capture rich multi-scale nuclear-to-cytoplasmic features while maintaining computational efficiency.
2. **Kolmogorov-Arnold Network (KAN) Head**: Replaces conventional Multi-Layer Perceptrons (MLPs) with learnable 1D B-spline activation functions on edges, achieving superior non-linear fitting and parameter interpretability.
3. **5-Fold Stratified Group Cross-Validation**: Formulates slide-level grouping (`StratifiedGroupKFold`) across 12,749 liquid-based cytology images to ensure strict zero-data-leakage across patient cases.
4. **Comprehensive Anti-Overfitting Defenses**: Incorporates Random Resized Crop, Random Erasing (Cutout), ColorJitter, Label Smoothing ($0.08$), AdamW Weight Decay ($2\times 10^{-4}$), and Early Stopping.
5. **Nuclear-Centric Explainable AI**: Validated through **Grad-CAM** saliency maps confirming that decisions are driven by cellular chromatin condensation and nuclear enlargement.

---

## 📊 Benchmark Results

All models were evaluated using **Stratified 5-Fold Group Cross-Validation** on the identical CCID dataset containing **12,749 cervical cell images** across **6 Bethesda diagnostic categories**.

### 1. Overall Model Comparison

| Backbone Architecture | Head | Accuracy (ACC) | Sensitivity (Recall) | Specificity (SPE) | Precision (PRE) | F1-Score | Macro AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RegNetY-16GF (Proposed)** | **KAN** | **0.9405** | **0.9152** | **0.9867** | **0.9212** | **0.9178** | **0.9839** |
| DenseNet-201 | KAN | 0.9360 | 0.9151 | 0.9859 | 0.9101 | 0.9122 | **0.9867** |
| EfficientNetV2-L | KAN | 0.9395 | 0.9180 | 0.9872 | 0.9140 | 0.9154 | 0.9788 |
| ConvNeXt-Large | KAN | 0.9266 | 0.9158 | 0.9849 | 0.8874 | 0.9002 | 0.9761 |
| Swin Transformer V2-B | KAN | 0.7723 | 0.7611 | 0.9525 | 0.7433 | 0.7395 | 0.9521 |

> **Highlight:** The proposed **RegNetY-16GF + KAN** model achieves the highest overall accuracy (**94.05%**), precision (**92.12%**), and F1-Score (**0.9178**), establishing state-of-the-art capability in automated cytological screening.

---

### 2. Per-Class Diagnostic Performance of RegNetY-16GF + KAN

| Class Label | Bethesda Diagnosis | Accuracy | Sensitivity (SEN) | Specificity (SPE) | Precision (PRE) | F1-Score | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0_Normal** | Negative for Intraepithelial Lesion / Malignancy | 0.9747 | 0.9781 | 0.9715 | 0.9706 | 0.9743 | 0.9920 |
| **1_LSIL** | Low-grade Squamous Intraepithelial Lesion | 0.9862 | 0.9098 | 0.9958 | 0.9642 | 0.9362 | 0.9898 |
| **2_HSIL** | High-grade Squamous Intraepithelial Lesion | 0.9651 | 0.8239 | 0.9779 | 0.7706 | 0.7963 | 0.9585 |
| **3_CIS** | Carcinoma In Situ | 0.9809 | 0.8660 | 0.9911 | 0.8969 | 0.8812 | 0.9812 |
| **4_SCC** | Squamous Cell Carcinoma | 0.9755 | 0.9389 | 0.9847 | 0.9389 | 0.9389 | 0.9865 |
| **5_Adeno** | Endocervical / Endometrial Adenocarcinoma | 0.9987 | 0.9744 | 0.9995 | 0.9858 | 0.9801 | 0.9953 |
| **Macro Average** | *Overall Diagnostic Average* | **0.9405** | **0.9152** | **0.9867** | **0.9212** | **0.9178** | **0.9839** |

---

## 📈 Visual Performance & Explainability

### Confusion Matrix & Multi-Class ROC Curves

<p align="center">
  <img src="assets/regnet_y_16gf_confusion_matrix.png" width="48%" alt="RegNetY-16GF Confusion Matrix"/>
  <img src="assets/regnet_y_16gf_roc_curve.png" width="48%" alt="RegNetY-16GF ROC Curve"/>
</p>
<p align="center">
  <em>Figure 2: Out-Of-Fold (OOF) Confusion Matrix (left) and Multi-Class One-vs-Rest ROC curves (right) for the RegNetY-16GF + KAN model.</em>
</p>

### Explainable AI (Grad-CAM) Visualizations

<p align="center">
  <img src="assets/gradcam/cam_000.png" width="85%" alt="Grad-CAM Sample 1"/>
</p>
<p align="center">
  <em>Figure 3: Grad-CAM attention maps highlighting precise diagnostic localization on abnormal cellular nuclei and hyperchromasia.</em>
</p>

---

## 📁 Repository Structure

```text
RegNetY16GF-KAN-Cervical-Cytology/
├── assets/                                 # 300 DPI figures, diagrams, and ROC curves
│   ├── proposed_regnet_kan_architecture.png
│   ├── proposed_regnet_kan_architecture.pdf
│   ├── regnet_y_16gf_confusion_matrix.png
│   ├── regnet_y_16gf_roc_curve.png
│   └── gradcam/                            # Sample Grad-CAM attention visualizations
├── data/
│   └── README.md                           # Dataset organization and slide grouping format
├── results/                                # Benchmark CSV reports and run configurations
│   ├── model_comparison_table.csv
│   ├── regnet_y_16gf_metrics.csv
│   └── run_config.json
├── scripts/
│   └── render_architecture_diagram.py      # Standalone 300 DPI architecture generator
├── src/
│   ├── models/
│   │   ├── kan.py                          # PyTorch-native B-spline KANLinear layer
│   │   ├── backbones.py                    # Feature backbone extractors (RegNet, DenseNet, etc.)
│   │   └── hybrid_kan.py                   # HybridBackboneKAN and FusionFiveBackboneKAN modules
│   └── utils/
│       ├── dataset.py                      # CCID Dataset loader with slide-level grouping
│       ├── transforms.py                   # Cytological data augmentations (Cutout, ColorJitter)
│       ├── metrics.py                      # Clinical evaluation metrics (ACC, SEN, SPE, PRE, F1, AUC)
│       ├── gradcam.py                      # Saliency engine for Explainable AI
│       └── visualize.py                    # Plotting utilities for Confusion Matrices & ROC Curves
├── train.py                                # Main 5-fold cross-validation training script
├── eval.py                                 # Single-image / Batch evaluation & Grad-CAM script
├── environment.yml                         # Conda environment definition
├── requirements.txt                        # Python dependencies
├── setup_git.bat                           # Quick GitHub initialization script
├── LICENSE                                 # MIT License
└── README.md                               # Repository documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

#### Using Pip & Virtualenv:
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/RegNetY16GF-KAN-Cervical-Cytology.git
cd RegNetY16GF-KAN-Cervical-Cytology

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install PyTorch 2.4.1 with CUDA 12.4 (recommended for Python 3.12 & NVIDIA A40 / Ampere / Ada GPUs):
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r requirements.txt
```

#### Using Conda:
```bash
conda env create -f environment.yml
conda activate regnet-kan-cytology
```

---

### 2. Dataset Setup

Organize your cervical cytology dataset under `data/train/` in class-specific folders:

```text
data/
└── train/
    ├── 0_Normal/
    ├── 1_LSIL/
    ├── 2_HSIL/
    ├── 3_CIS/
    ├── 4_SCC/
    └── 5_Adeno/
```

*Note: For detailed naming conventions to avoid patient-level data leakage, see [data/README.md](data/README.md).*

---

### 3. Training

#### Train Proposed RegNetY-16GF + KAN Model (5-Fold Stratified Group CV):
```bash
python train.py --model regnet_y_16gf --batch_size 16 --epochs 35 --lr 2e-4 --amp
```

#### Train on Multi-GPU Setup (e.g. 2x NVIDIA A40):
The training script automatically detects and utilizes all available GPUs with `torch.nn.DataParallel` and mixed precision (AMP FP16):
```bash
python train.py --model regnet_y_16gf --batch_size 32 --workers 8 --amp
```

#### Train All 5 Benchmark Models:
```bash
python train.py --model all --epochs 35 --batch_size 16
```

---

### 4. Inference & Grad-CAM Visualization

#### Single Image Prediction with Saliency Map:
```bash
python eval.py --weights results/checkpoints/regnet_y_16gf_fold1.pt --image sample_cell.png --gradcam
```

#### Batch Inference on a Directory:
```bash
python eval.py --weights results/checkpoints/regnet_y_16gf_fold1.pt --image_dir path/to/test_images --output_dir eval_results
```

---

### 5. Regenerating Publication Architecture Diagram

To reproduce the vector PDF and 300 DPI high-resolution architecture diagram:
```bash
python scripts/render_architecture_diagram.py
```

---

## 📝 Citation

If you find this code or research useful in your work, please cite:

```bibtex
@inproceedings{alperen2026regnetkan,
  title     = {Hybrid RegNetY-16GF and Kolmogorov-Arnold Networks for Multi-Class Cervical Cytology Classification and Nuclear Explainability},
  author    = {Alperen and Asl{\i}},
  booktitle = {Proceedings of the IEEE International Conference on Electrical, Communication and Energy Research (ICECER)},
  year      = {2026}
}
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
