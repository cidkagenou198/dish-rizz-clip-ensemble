# 🍽️ Match the Dish to its Rizz

> **Text-to-image food retrieval (R-precision) using an ensemble of SigLIP, EVA02 and DINOv2 features. Kaggle competition solution.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/)
[![License: Academic / Educational](https://img.shields.io/badge/License-Competition_Fair_Use-yellow.svg)](DATA_TERMS.txt)

---

## 🏷️ Topics & Tags
`kaggle` · `image-retrieval` · `siglip` · `pytorch` · `multimodal` · `dinov2` · `eva02` · `optimal-transport` · `sinkhorn`

---

## 🏆 Leaderboard & Performance

| Benchmark / Evaluation | Metric | Score | Notes |
| :--- | :--- | :--- | :--- |
| **Validation R-Precision (Top-20)** | R-Precision | **0.865+** | Full Multi-Backbone Ensemble |
| **Quad Ensemble (SigLIP + EVA02 + DINOv2 + ConvNeXt)** | Top-1 Accuracy | **88.2%** | With Sinkhorn Optimal Transport |
| **Gallery Manifold Diffusion** | Boost over baseline | **+2.4%** | Nearest-neighbor visual smoothing |

---

## 🔬 Method Summary

The task requires retrieving and matching fine-grained gallery food images corresponding to ambiguous or natural-language dish descriptions across **101 culinary categories**.

```
┌─────────────────────────┐       ┌────────────────────────┐
│ Text Queries & Descs    │ ────> │ NLP Query Mapper       │ ────┐
│ (Paraphrases & Aliases) │       │ (Regex + Paraphrases)  │     │
└─────────────────────────┘       └────────────────────────┘     │
                                                                 ▼
┌─────────────────────────┐       ┌────────────────────────┐   ┌───────────────────────────────┐   ┌─────────────────────────┐
│ Gallery Food Images     │ ────> │ Vision Backbone Models │──>│ Ensemble & Re-ranking         │──>│ Ranked Dish Predictions │
│ (101 Food Classes)      │       │ (SigLIP, EVA02, DINOv2)│   │ (Sinkhorn + Manifold Diff.)   │   │ (Submission Top-20)     │
└─────────────────────────┘       └────────────────────────┘   └───────────────────────────────┘   └─────────────────────────┘
```

### 1. Vision Feature Ensembling
- **Foundation Backbones**:
  - **SigLIP** (Sigmoid loss Vision-Language Pretraining) for strong text-image alignment.
  - **EVA02** & **DINOv2** for high-fidelity self-supervised visual representations and texture understanding.
  - **ConvNeXt-Base** & **Swin-B** fine-tuned on target dish distributions with Mixup, CutMix, RandAugment, and Label Smoothing.
- **Test-Time Augmentation (TTA)**: 5-crop and horizontal flip averaging.

### 2. NLP Query Mapping (`src/query_mapper.py`)
- Unicode NFKD normalization and accent stripping.
- Rule-based regex token matching and length-prioritized substring mapping across canonical dish names, alternative aliases, and rich paraphrases from `category_metadata.csv`.

### 3. Post-Processing & Optimal Transport Re-ranking (`src/reranker.py`)
- **Sinkhorn-Knopp Matrix Balancing**: Formulates retrieval as an optimal transport problem to enforce equitable class distributions across the gallery, preventing over-allocation to dominant classes.
- **Gallery Manifold Diffusion**: Propagates confidence scores through the visual nearest-neighbor graph ($k$-NN) of the gallery features to smooth prediction noise.

---

## 📂 Repository Structure

```text
├── notebooks/
│   └── solution.ipynb             # Interactive walk-through of the end-to-end solution
├── src/
│   ├── dataset.py                 # PyTorch Dataset loaders and augmentations
│   ├── finetune.py                # Backbone fine-tuning script
│   ├── feature_extractor.py       # Feature extraction and caching utilities
│   ├── query_mapper.py            # NLP text query parser and category mapper
│   ├── reranker.py                # Sinkhorn-Knopp balancing & gallery manifold diffusion
│   ├── metrics.py                 # Evaluation metrics (R-Precision, mAP, Recall@K)
│   ├── evaluate_models.py         # Validation evaluation harness
│   ├── ensemble_evaluation.py     # Multi-model ensemble evaluation
│   ├── tri_ensemble_evaluation.py # 3-way ensemble benchmarking
│   ├── quad_ensemble_evaluation.py# 4-way ensemble benchmarking
│   ├── run_finetuned_submission.py# Complete fine-tune + submission generation pipeline
│   ├── generate_submission.py     # Submission CSV formatting
│   └── validate_submission.py     # Schema and submission integrity validator
├── category_metadata.csv          # 101 category IDs, descriptions, and paraphrases
├── DATA_TERMS.txt                 # Competition terms and data conditions
├── requirements.txt               # Dependencies
└── .gitignore                     # Ignores large checkpoints, images, and feature caches
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/cidkagenou198/dish-rizz-clip-ensemble.git
cd dish-rizz-clip-ensemble
pip install -r requirements.txt
```

### 2. Interactive Notebook
Launch Jupyter to explore the methodology and pipeline:
```bash
jupyter notebook notebooks/solution.ipynb
```

### 3. Fine-Tuning & Submission Pipeline
```bash
# Fine-tune backbone
python src/finetune.py --epochs 25 --batch-size 16

# Run full ensemble and generate submission
python src/run_finetuned_submission.py

# Verify submission formatting
python src/validate_submission.py submission.csv
```

---

## 📜 Terms & Fair Use
Images and labels used in the competition are governed by [`DATA_TERMS.txt`](DATA_TERMS.txt). All code is provided for research and educational purposes.
