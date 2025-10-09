# Modeling Handoff — Week 6  
*Prepared by: Gisselle (C)*  
*Date: October 9, 2025*

---

## 1. Purpose of This Document

This file provides the full technical record and interpretation of the **modeling phase** for the UCI Student Performance dataset. It serves as an internal handoff and reproducibility guide for teammates (especially Francisco) writing the “Model Selection” and “Model Analysis” sections of `report.md`.

The summarized results already appear in `report.md`; this document contains **all model parameters, file paths, outputs, and explanatory notes** to prevent ambiguity or misinterpretation.

---

## 2. Clean Data Used for Modeling

- Dataset: `data/clean/student_clean_classification_noG1G2.csv`
- Rows: 1,044  
- Target variable: `performance_level` (derived from G3)  
  - **Low:** G3 < 10  
  - **Medium:** 10 ≤ G3 < 15  
  - **High:** G3 ≥ 15  
- G1, G2, and G3 were **removed from the feature set** to prevent data leakage.  
- Train/test split: **80/20**, stratified by `performance_level`, `random_state=42`.

---

## 3. Models Trained

| Model | Library & Parameters | Notes |
|--------|-----------------------|--------|
| **Logistic Regression (OvR)** | `StandardScaler(with_mean=False)`, `class_weight='balanced'`, `multi_class='ovr'`, `max_iter=2000` | Baseline linear classifier for interpretability. |
| **Random Forest Classifier** | `n_estimators=300`, `min_samples_split=4`, `class_weight='balanced'`, `random_state=42` | Captures nonlinear effects and variable interactions. |
| **SVM (RBF kernel)** | `C=2.0`, `gamma='scale'`, `class_weight='balanced'`, `kernel='rbf'` | Margin-based classifier for comparison; less suited for mixed categorical data. |

---

## 4. Performance Summary

| Model | Accuracy | Macro-F1 |
|--------|-----------|-----------|
| Logistic Regression (no leak) | **0.507** | **0.504** |
| **Random Forest** | **0.636** | **0.502** |
| SVM (RBF kernel) | **0.474** | **0.479** |

**Interpretation:**  
Random Forest performed best overall, confirming that nonlinear relationships between variables (e.g., between parental education, absences, and failures) are meaningful.  
Logistic Regression achieved similar macro-F1 but lower accuracy, showing linear separability limits.  
SVM underperformed due to categorical dummy-variable structure and overlapping class boundaries.

---

## 5. File Artifacts

All outputs generated via the final export cell in `notebooks/02_Modeling.ipynb`.

| Type | File Path | Description |
|------|------------|--------------|
| **Metrics Table** | `data/model_outputs/model_metrics_summary.csv` | Accuracy & macro-F1 for all models. |
| **Confusion Matrices** | `data/model_outputs/confusion_matrix_logistic_regression.png`<br>`data/model_outputs/confusion_matrix_random_forest.png`<br>`data/model_outputs/confusion_matrix_svm_rbf.png` | Visual confusion matrices for each model. |
| **Random Forest Importances (Table)** | `data/model_outputs/rf_feature_importances.csv` | Full ranked feature importance values. |
| **Random Forest Importances (Plot)** | `data/model_outputs/rf_feature_importances_top15.png` | Bar plot of top 15 most important features. |
| **Logistic Regression Odds Ratios** | `data/model_outputs/logreg_odds_ratios_High.csv` | Odds ratios for “High” class vs. others. |

All files saved automatically when running the “Handoff Summary” cell.

---

## 6. Key Findings by Model

### Logistic Regression (no leakage)
**Positive predictors (higher odds of High performance):**
- `higher_yes` — desire for higher education (~3× odds)
- `Medu`, `Fedu` — parental education levels
- `studytime`
- `internet_yes`
- `Mjob_services`, `Fjob_teacher`, `Fjob_health`

**Negative predictors (lower odds of High performance):**
- `failures`
- `absences`
- `schoolsup_yes`
- `Walc`, `Dalc` (alcohol use)
- `goout`
- `health` (lower values)

### Random Forest (top importances)
| Rank | Feature | Explanation |
|------|----------|-------------|
| 1 | `failures` | Past academic failure is strongest predictor of poor performance. |
| 2 | `absences` | Attendance directly affects outcomes. |
| 3 | `Medu` | Mother’s education strongly correlated with better performance. |
| 4 | `age` | Older students (repeaters) often perform worse. |
| 5 | `goout` | Excessive social outings reduce study time. |
| 6 | `Fedu` | Father’s education also significant. |
| 7–10 | `Walc`, `health`, `freetime`, `studytime` | Lifestyle and work habits influence consistency. |

### Cross-model consistency
Both models highlight **engagement (studytime, absences, failures)** and **family background (Medu, Fedu)** as central predictors. Alcohol use (`Dalc`, `Walc`) and support needs (`schoolsup_yes`) are negative indicators.

---

## 7. Confusion Matrix Patterns

| Observation | Implication |
|--------------|-------------|
| “Medium” class has the most predictions | Overlap with both Low and High groups makes separation difficult. |
| “High” performance recall ~0.71 in SVM, lower elsewhere | High performers are fewer and harder to isolate. |
| Random Forest shows stronger diagonals overall | Suggests more confident predictions in all three categories. |

---

## 8. Limitations & Caveats

- Models are **correlational**, not causal.  
- “Medium” performance overlaps with both extremes → expected misclassifications.  
- **Self-reported variables** (e.g., alcohol use, study time) may contain bias.  
- Random Forest importances are biased toward variables with more split points.  
- No hyperparameter tuning yet (baseline only).  
- Class imbalance persists (Medium = majority).

---

## 9. Recommendations

- Implement **SMOTE or class weighting** for improved recall on minority classes.  
- Conduct **hyperparameter tuning** (GridSearchCV) for Random Forest and SVM.  
- Explore **SHAP or permutation importances** for more interpretable non-linear effects.  
- Add a **pass/fail binary model** variant for simplified classification comparison.  

---

## 10. Reproducibility

To reproduce these outputs:

1. Run `notebooks/01_preprocess_clean.ipynb` to regenerate clean data.  
2. Run `notebooks/02_Modeling.ipynb` end-to-end.  
3. All results will appear automatically in `data/model_outputs/`.  

Ensure `.gitignore` allows this folder to sync:

!data/model_outputs/
!data/model_outputs/*

---

## 11. Do / Don’t Summary

✅ **Do**
- Reference Logistic Regression for *direction* (↑/↓ odds).  
- Reference Random Forest for *priority* (which factors matter most).  
- Use phrasing like “associated with” instead of “causes.”  
- Cite file paths exactly as listed above.

🚫 **Don’t**
- Reintroduce `G1`, `G2`, or `G3` as predictors (leakage risk).  
- Treat correlations as causations.  
- Drop context on the Medium-class overlap when describing results.

---

## 12. Figure Captions (for `report.md`)

**Figure X.** Confusion matrices for Logistic Regression, Random Forest, and SVM (RBF). Random Forest shows the clearest diagonal pattern, especially for the Medium class.

**Figure Y.** Top 15 feature importances from Random Forest. *Failures* and *absences* dominate, followed by *parental education* and *studytime*.

---

## 13. Quick Reuse Note

If future teams extend this project:
- Keep this document as the technical appendix.
- Update results by rerunning the export cell in `02_Modeling.ipynb`.
- Maintain versioning via Git commits for each modeling iteration.

---

**End of modeling_handoff.md**

