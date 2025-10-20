# Final Project – AAI 500 (Team 4)

## Team Members
- **Tavia Wooten** – Initial Data Cleaning & Dataset Merging  
- **Francisco Monarrez Félix** – EDA, Report Writing, and Slide Design  
- **Noslinn Gisselle Tosta** – Data Preparation, Modeling, Tuning, and Documentation  

---

## Project Overview
This project performs an end-to-end statistical and machine-learning analysis using the  
**UCI Student Performance Dataset** ([UCI Repository Link](https://archive.ics.uci.edu/dataset/320/student+performance)).

### Objectives
- Identify key **demographic, social, and academic factors** associated with student performance.  
- Build and tune **classification models** to predict overall performance level (Low, Medium, High).  
- Interpret model outputs to understand which factors most influence academic outcomes.

---

## Repository Structure

Final-Project-Team-4-AAI500/
├── data/
│   ├── student_clean.csv   # Original dataset (from Tavia - kept for transparency)
│   ├── raw/                # Original datasets (student-mat.csv, student-por.csv)
│   ├── clean/              # Cleaned & merged datasets (classification-ready)
│   ├── model_outputs/      # Exported metrics, confusion matrices, and feature importances
│   └── _archive/           # Old or experimental files (optional)
│
├── docs/
│   └── modeling_handoff.md         # Full technical modeling appendix│
│
├── notebooks/
│   ├── 01_preprocess_clean.ipynb   # Combines, cleans, and encodes data
│   ├── 02_Modeling.ipynb           # Baseline models (LogReg, RF, SVM)
│   ├── final_models.ipynb          # Hyperparameter tuning + final model results
│   └── encoding_final.ipynb        # Early encoding experiment (from Tavia - kept for transparency)
│
├── src/
│   ├── encoding_logic.py           # Helper functions for preprocessing
│   └── final_models.py             # Final tuned models (LogReg, RF, SVM)
│
├── report.md                       # Final project report (main deliverable)
├── README.md                       # Overview and instructions
└── LICENSE

---

## Modeling Summary
Three models were trained and tuned on the cleaned dataset (`student_clean_classification_noG1G2.csv`)
using **5-fold cross-validation** and an **80/20 train-test split**.

| Model | Accuracy | Macro-F1 |
|--------|-----------|-----------|
| Logistic Regression (tuned) | 0.589 | 0.536 |
| **Random Forest (tuned)** | **0.670** | **0.623** |
| SVM (RBF kernel, tuned) | 0.536 | 0.533 |

- **Top Positive Predictors:** Study time, parental education, higher-education aspiration  
- **Top Negative Predictors:** Absences, alcohol use, prior failures  

Full details and visualizations are available in `docs/modeling_handoff.md`.

---


## Instructions for Reproduction
1. **Clone this repository**  
   ```bash
   git clone <repo-link>
   cd Final-Project-Team-4-AAI500
   ```

2. **To reproduce the tuned model results using the Python script**, run:  
   ```bash
   python src/final_models.py
   ```
   This will:
   - Load the cleaned dataset (`data/clean/student_clean_classification_noG1G2.csv`)
   - Train tuned Logistic Regression, Random Forest, and SVM (RBF) models
   - Save metrics, confusion matrices, and feature importances to `data/model_outputs/`

3. **For interactive analysis or visuals**, open these notebooks in JupyterLab or VS Code:  
   - `notebooks/01_preprocess_clean.ipynb` → Data cleaning and encoding  
   - `notebooks/02_Modeling.ipynb` → Baseline models (LogReg, RF, SVM)  
   - `notebooks/final_models.ipynb` → Hyperparameter tuning + final results  

4. **Outputs**  
   All generated files — metrics tables, confusion matrices, and feature importance charts —  
   will be saved automatically to:  
   ```
   data/model_outputs/
   ```

5. **Documentation**  
   - See `report.md` for the written analysis.  
   - See `docs/modeling_handoff.md` for the full technical appendix (parameters, code, and reasoning).  

---

## Notes

- The dataset merges both Math and Portuguese student records.  
- Grades (`G1`, `G2`, `G3`) were **excluded from predictors** to prevent target leakage.  
- **No imputation** was performed — incomplete rows were dropped to avoid data leakage.  
- Modeling results are **correlational**, not causal.  
- The repository follows **PEP 8** standards and uses **stratified train/test splits** for reproducibility.  

---

## University of San Diego – MS Applied Artificial Intelligence, Fall 2025
**AAI 500 – Probability and Statistics for Artificial Intelligence**  
Team 4 | Final Project Submission  

---

### What Changed
| Old | New |
|------|------|
| Mentioned `project_notebook.ipynb` | Removed (archived instead) |
| Mentioned `figures/` | Replaced with `/data/model_outputs/` |
| General objectives | Updated to match final research question |
| Repo structure | Aligned with actual project folders |
| Added | Reproducible script instructions + interactive notebook sequence |
| Updated | Notes to include no-imputation policy and data leakage prevention |

---
