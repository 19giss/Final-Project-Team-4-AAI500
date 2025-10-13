# Final Project Report (Updated October 12)

## Introduction
Our team selected the **UCI Student Performance dataset**, which combines information from two secondary school courses (Math and Portuguese). The dataset includes demographic, social, and academic attributes of students, with final grade (G3) serving as the target variable.  

The objective of our analysis is to explore factors that influence student performance and to establish predictive baselines. Specifically, we aim to answer:  
1. Which demographic, social, and academic factors are most strongly associated with final course performance (G3)?  
2. Can classification models reasonably predict overall performance levels (Low, Medium, High)?  

These questions guided our cleaning, exploratory data analysis (EDA), and modeling steps.

---

## Data Cleaning / Preparation
The cleaning process followed these main steps:
- Imported both datasets (`student-mat.csv` and `student-por.csv`).
- Inspected the data (`.info()` and `.head()`) to understand structure and spot inconsistencies.
- Saved copies of raw datasets for reference.
- Merged the two datasets into a single file (`merged_student.csv`).
- Checked for missing values and removed incomplete rows (`dropna()`).
- Checked for duplicates and removed them.
- Saved the cleaned dataset as `student_clean.csv` for further analysis.

The final cleaned dataset contains **1,044 rows × 34 columns**.  
No imputation was performed to avoid data leakage; instead, incomplete records were dropped prior to splitting.

---

## Exploratory Data Analysis (EDA)
We examined both numeric and categorical variables to understand the dataset before modeling.

### Numeric variables
- Most students were between 16 and 18 years old.  
- Final grade (G3) distribution was skewed toward the middle range (10–15 out of 20).  
- Absences were highly skewed, with most students below 10 and a few exceeding 50.  
- Mean final grade (G3) ≈ 11.9 (std ≈ 3.2).  

### Categorical variables
- Slightly more female than male students.  
- Most lived in urban areas and had internet access.  
- A majority intended to pursue higher education.  

### Correlations
- Strong positive correlations between G1, G2, and G3.  
- Moderate positive correlation between parental education (`Medu`, `Fedu`) and student performance.  

**Summary:**  
EDA suggested that **study habits, attendance, and family background** are likely key predictors of performance.

---

## Model Selection (Baseline)
We trained three baseline classifiers — **Logistic Regression**, **Random Forest**, and **SVM (RBF)** — to predict `performance_level` (Low <10, Medium 10–14, High ≥15).  
Grades G1, G2, and G3 were excluded from predictors to prevent leakage.

| Model | Accuracy | Macro-F1 |
|--------|-----------|-----------|
| Logistic Regression (no leak) | 0.507 | 0.504 |
| **Random Forest** | **0.636** | **0.502** |
| SVM (RBF kernel) | 0.474 | 0.479 |

**Training details**
- 80/20 stratified split (`random_state=42`)  
- Logistic Regression: scaled with `StandardScaler(with_mean=False)`, `class_weight='balanced'`  
- Random Forest: `n_estimators=300`, `min_samples_split=4`, `class_weight='balanced'`  
- SVM: RBF kernel, `C=2.0`, `gamma='scale'`, `class_weight='balanced'`

---

## Model Tuning (5-Fold Cross-Validation)
To improve generalization, we performed **GridSearchCV** (5-fold stratified, macro-F1 scoring) on all three models.

| Model | Accuracy | Macro-F1 |
|--------|-----------|-----------|
| Logistic Regression (tuned) | 0.589 | 0.536 |
| **Random Forest (tuned)** | **0.670** | **0.623** |
| SVM (RBF, tuned) | 0.536 | 0.533 |

**Best hyperparameters**
- Logistic Regression: `C=0.25`, `penalty='l2'`, `solver='liblinear'`
- Random Forest: `n_estimators=200`, `max_features='log2'`, `min_samples_split=2`, `min_samples_leaf=4`
- SVM: `C=4.0`, `gamma=0.01`

**Results summary**
- **Random Forest** improved most after tuning, confirming that nonlinear interactions among social, demographic, and academic features are important.  
- **Logistic Regression** offered the most interpretable coefficients but slightly lower predictive power.  
- **SVM** performed moderately well but struggled with class imbalance, especially for the “Medium” category.

All confusion matrices, feature importances, and metrics were exported to  
`data/model_outputs/`.

---

## Model Analysis

**Positive predictors of high performance**
- Desire for higher education (`higher_yes`)
- Parental education (`Medu`, `Fedu`)
- Study time
- Internet access

**Negative predictors**
- Previous failures
- Absences
- School support needs (`schoolsup_yes`)
- Alcohol consumption (`Walc`, `Dalc`)
- High social activity (`goout`)

**Interpretation:**  
The **Random Forest** model achieved the best performance, with **accuracy = 0.67** and **macro-F1 = 0.62**, suggesting it generalized well across categories.  
Behavioral and family-related factors consistently influenced student outcomes, supporting the hypothesis that engagement and environment drive academic success.  
The “Medium” group remained the most challenging to classify due to overlap with both higher and lower performers.

---

## Limitations
- Relationships are **correlational**, not causal.  
- Potential bias from **self-reported variables** (e.g., alcohol use, study time).  
- **Class imbalance** (Medium dominates).  
- **No imputation** was performed to prevent data leakage; incomplete records were removed.  
- Random Forest importances may favor continuous variables with more unique values.

---

## Conclusion & Recommendations
Model tuning significantly improved predictive accuracy, with the Random Forest model achieving the highest performance after optimization.  
Results highlight the impact of **study engagement**, **parental education**, and **attendance** on academic success.  

**Recommendations**
1. Future models could incorporate **balancing methods** (e.g., SMOTE) to improve recall for minority classes.  
2. Adding **SHAP value interpretation** can enhance explainability.  
3. Testing **gradient-boosted models** (XGBoost, CatBoost) may further improve performance without explicit imputation.  

**Final takeaway:**  
The project demonstrates that student performance can be predicted with moderate accuracy using demographic and behavioral data, emphasizing the strong influence of engagement and family support.
