# Hierarchical Two-Stage Classifier for Rheumatic Disease Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Journal: Computers in Biology and Medicine](https://img.shields.io/badge/Journal-Computers%20in%20Biology%20%26%20Medicine-green)](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine)

A hierarchical two-stage machine learning pipeline for multi-class classification of **7 rheumatic and autoimmune diseases** from routine serological data.  
**Novel contribution:** confidence-gated boundary resolver that fixes the AS (Ankylosing Spondylitis) misclassification gap identified in the existing literature.

> **Paper:** *Hierarchical Two-Stage Machine Learning for Rheumatic and Autoimmune Disease Classification: Resolving the Ankylosing Spondylitis Diagnostic Gap*  
> **Authors:** [Monjurul Islam], Barshon Sen  
> **Institution:** Rajshahi University of Engineering and Technology (RUET), Bangladesh  
> **Status:** Submitted to *Computers in Biology and Medicine* (Elsevier, IF 7.7, Q1)

---

## The Problem

Ankylosing Spondylitis (AS) has an average diagnostic delay of **8–10 years** in clinical practice.  
The best published baseline on this dataset achieved only **57.6% AS recall** — 25.6% of AS patients were misclassified as RA.

Single-stage classifiers cannot simultaneously optimise AS recall and RA recall because the two diseases share overlapping serological profiles (HLA-B27, RF, Anti-CCP). Lowering the AS threshold improves AS recall but destroys RA recall.

## Our Solution

A **hierarchical two-stage architecture**:

```
Raw Data (12,085 patients × 15 features)
        ↓
  Preprocessing + HLA-B27 Soft Imputation
        ↓
  44-Feature Clinical Fingerprint Engineering
        ↓
┌─────────────────────────────────────┐
│  Stage 1 — Global Balanced Ensemble │
│  XGBoost (60%) + DART (10%) +       │
│  TabNet (30%) — no AS bias          │
└──────────────┬──────────────────────┘
               ↓
     Confidence ≥ 0.48?
      ╱               ╲
   YES                  NO (1.9% of cases)
    ↓                    ↓
Stage 1              Stage 2 — Boundary Resolver
prediction           AS+RA+PsA specialist model
                     (AS weight boosted ×2.5)
      ╲               ╱
       ↓             ↓
     Final Prediction (7 classes)
```

---

## Dataset

- **Source:** Mahdi et al. (2025) — *Data in Brief*, Elsevier
- **Repository:** [Harvard Dataverse](https://doi.org/10.7910/DVN/VM4OR3)
- **Patients:** 12,085 de-identified records
- **Classes:** Ankylosing Spondylitis · Normal · Psoriatic Arthritis · Reactive Arthritis · Rheumatoid Arthritis · Sjögren's Syndrome · Systemic Lupus Erythematosus
- **Features:** 15 raw (2 demographic + 13 serological) → 44 engineered

Download the dataset from Harvard Dataverse and place it as:
```
data/Rheumatic and Autoimmune Disease Dataset.csv
```

---

## Project Layout

```
rheumatic-classifier/
├── data/
│   └── README_data.md          Instructions to download dataset
├── src/
│   ├── preprocess.py           Base preprocessing + HLA-B27 imputation
│   ├── feature_engineering.py  44-feature clinical fingerprint pipeline
│   ├── train_stage1.py         Global balanced ensemble (XGB+DART+TabNet)
│   ├── train_stage2.py         Boundary resolver (AS+RA+PsA specialist)
│   ├── ensemble.py             Soft-voting + confidence routing
│   ├── evaluate.py             Full evaluation (journal-format tables)
│   ├── predict.py              Inference on new data
│   └── plot_results.py         Publication figures (300 DPI)
├── models/                     Saved model files (after training)
├── figures/
│   ├── figure_architecture.png     Pipeline architecture diagram
│   ├── figure_feature_importance.png
│   ├── figure_model_comparison.png
│   ├── confusion_matrices_all_models.png
│   └── confusion_matrix_final_model.png
├── results/
│   ├── metrics_final.json      All evaluation metrics
│   └── test_predictions.csv    Per-sample predictions + probabilities
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Results

| Model | AS Recall | AS F1 | RA Recall | Macro F1 | Accuracy |
|---|---|---|---|---|---|
| Original RF (Baseline) | 57.6% | 65.9% | 91.4% | 83.8% | 84.2% |
| v1: XGB + Threshold | 90.8% | 70.6% | 78.3% | 83.2% | 82.2% |
| v2: XGB + OvR Ensemble | 90.8% | 70.7% | 70.2% | 83.6% | 82.4% |
| v3: Grand Ensemble | 92.7% | 71.1% | 67.4% | 84.2% | 82.7% |
| **v4 Hierarchical (FINAL)** | **65.6%** | **66.7%** | **85.6%** | **84.5%** | **84.0%** |

**Key metrics (final model):**
- Cohen Kappa: **0.808**
- Matthews CC: **0.809**
- SLE F1: **98.9%** (perfect complement signature)
- Stage 2 activated: **1.9%** of test samples only

**5-fold CV:** Macro F1 = 83.06% ± 0.66% (95% CI: 82.48–83.64%)

---

## Installation

```bash
git clone https://github.com/[your-username]/rheumatic-classifier.git
cd rheumatic-classifier
pip install -r requirements.txt
```

---

## Usage

```bash
# 1. Preprocess + engineer features (~1 min)
python src/feature_engineering.py

# 2. Train Stage 1 global ensemble + Optuna tuning (~10–15 min)
python src/train_stage1.py

# 3. Train Stage 2 boundary resolver (~3 min)
python src/train_stage2.py

# 4. Run full evaluation + save results
python src/evaluate.py

# 5. Generate all publication figures (300 DPI)
python src/plot_results.py

# 6. Predict on new data
python src/predict.py --input your_data.csv --output predictions.csv
```

---

## Novelty

1. **Hierarchical two-stage architecture** — first confidence-gated routing in rheumatic disease classification
2. **HLA-B27 probabilistic soft imputation** — continuous confidence score as feature instead of hard binary fill
3. **44-feature clinical fingerprint set** — ESR corridor, RF/CCP RA-distance, HLA-B27 interaction terms, 3 composite AS scores
4. **Irreducible overlap quantification** — 31 AS boundary patients formally profiled; establishes hard floor and its clinical cause
5. **Optuna-tuned reproducible pipeline** — seed=42 throughout, fully reproducible

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{islam2026hierarchical,
  title     = {Hierarchical Two-Stage Machine Learning for Rheumatic and 
               Autoimmune Disease Classification: Resolving the 
               Ankylosing Spondylitis Diagnostic Gap},
  author    = {Islam, Monjurul and Sen, Barshon},
  journal   = {Computers in Biology and Medicine},
  year      = {2026},
  publisher = {Elsevier}
}
```

Also cite the original dataset:

```bibtex
@article{mahdi2025dataset,
  title   = {Diagnosis of rheumatic and autoimmune diseases dataset},
  author  = {Mahdi, Mohammed Fadhil and Jahani, Arezoo and Abd, Dhafar Hamed},
  journal = {Data in Brief},
  volume  = {60},
  pages   = {111623},
  year    = {2025},
  doi     = {10.1016/j.dib.2025.111623}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Dataset: Mahdi et al. (2025), Harvard Dataverse (DOI: 10.7910/DVN/VM4OR3)  
Supervisor: Asst. Prof. Barshon Sen, RUET
# rheumatic-classifier
