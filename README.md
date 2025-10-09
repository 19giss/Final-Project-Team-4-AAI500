# Final Project – AAI 500 (Team 4)

## 👥 Team Members
- **Tavia Wooten** – Data Cleaning & Preparation  
- **Francisco Monarrez Félix** – EDA & Visualization  
- **Noslinn Gisselle Tosta** – Modeling & Metrics  

---

## 📘 Project Overview
This project performs an end-to-end statistical and machine-learning analysis using the  
**UCI Student Performance Dataset** ([UCI Repository Link](https://archive.ics.uci.edu/dataset/320/student+performance)).

### Objectives
- Identify key **demographic, social, and academic factors** associated with student performance.  
- Build **baseline classification models** to predict overall performance level (Low, Medium, High).  
- Interpret model outputs to understand which factors most influence academic outcomes.

---

## 🗂️ Repository Structure

Final-Project-Team-4-AAI500/
├── data/
│   ├── raw/                # Original datasets (student-mat.csv, student-por.csv)
│   ├── clean/              # Cleaned & merged datasets (classification-ready)
│   ├── model_outputs/      # Exported metrics, confusion matrices, and feature importances
│   └── _archive/           # Old or experimental files (optional)
│
├── notebooks/
│   ├── 01_preprocess_clean.ipynb   # Combines, cleans, and encodes data
│   ├── 02_Modeling.ipynb           # Baseline models (LogReg, RF, SVM)
│   └── encoding_final.ipynb        # Early encoding experiment (kept for transparency)
│
├── src/
│   └── encoding_logic.py           # Helper functions for preprocessing
│
├── docs/
│   └── modeling_handoff.md         # Full technical modeling appendix (for internal handoff)
│
├── report.md                       # Project report (final submission)
├── README.md                       # Overview and instructions
└── LICENSE

---

## 🧠 Modeling Summary
Three baseline models were trained on the cleaned dataset (`student_clean_classification_noG1G2.csv`):

| Model | Accuracy | Macro-F1 |
|--------|-----------|-----------|
| Logistic Regression (no leak) | 0.507 | 0.504 |
| **Random Forest** | **0.636** | **0.502** |
| SVM (RBF kernel) | 0.474 | 0.479 |

- **Key positive predictors:** higher education aspiration, parental education, study time, internet access.  
- **Key negative predictors:** failures, absences, school support needs, alcohol use.

Full details and visualizations are in `docs/modeling_handoff.md`.

---

## ⚙️ Instructions for Reproduction
1. Clone this repository  
   ```bash
   git clone <repo-link>
   cd Final-Project-Team-4-AAI500

2. Open JupyterLab or VS Code and run notebooks in order:

- notebooks/01_preprocess_clean.ipynb

- notebooks/02_Modeling.ipynb

3. Outputs (metrics, confusion matrices, feature importances) will be saved to:

   data/model_outputs/

4. See report.md for written analysis and docs/modeling_handoff.md for the full technical appendix.


## Notes

The dataset merges both Math and Portuguese student records.

Grades (G1, G2, G3) are excluded from predictors to prevent target leakage.

Modeling results are correlational, not causal.

This repository follows PEP 8 style and uses stratified splits for reproducibility.


## University of San Diego – MS Applied Artificial Intelligence, Fall 2025
Team 4 | AAI 500 – Foundations of Data Science



---

### 🧩 What Changed
| Old | New |
|------|------|
| Mentioned `project_notebook.ipynb` | Removed (archived instead) |
| Mentioned `figures/` | Replaced with `/data/model_outputs/` |
| General objectives | Updated to match final research question |
| Repo structure | Aligned with your real current folders |
| Added | Modeling summary table + clear reproduction steps |

---
