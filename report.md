# Final Project Report (Draft - Updated September 29)

## Introduction
Our team selected the **UCI Student Performance dataset**, which combines information from two secondary school courses (Math and Portuguese). The dataset includes demographic, social, and academic attributes of students, with final grade (G3) serving as the target variable.  

The objective of our analysis is to explore factors that influence student performance and to establish predictive baselines. Specifically, we aim to answer:  
1. Which demographic, social, and academic factors are most strongly associated with final course performance (G3)?  
2. Can baseline models (e.g., regression and classification) reasonably predict final grades or pass/fail status?  

These questions will guide our cleaning, exploratory data analysis (EDA), and modeling steps in the coming weeks.

## Data Cleaning / Preparation
The cleaning process followed these main steps:
- Imported both datasets (`student-mat.csv` and `student-por.csv`).
- Inspected the data (`.info()` and `.head()`) to understand structure and spot inconsistencies.
- Saved copies of raw datasets for reference.
- Merged the two datasets into a single file (`merged_student.csv`).
- Checked for missing values and created a cleaned version with `dropna()`.
- Checked for duplicates and removed them.
- Saved the cleaned dataset as `student_clean.csv` for further analysis.

The final cleaned dataset contains **1,044 rows × 34 columns**.

## Exploratory Data Analysis (EDA)
We examined both numeric and categorical variables to understand the dataset before modeling.

### Numeric variables
- Most students were between 16 and 18 years old.  
- The final grade (G3) distribution was skewed toward the middle range (10–15 out of 20), with relatively few students achieving very low (0–5) or very high (18–20) scores.  
- Absences were highly skewed, with the majority of students having fewer than 10, but some outliers exceeded 50.  

Additional descriptive statistics:  
- Average student age was ~16.7 years (std ≈ 1.2).  
- Average absences were ~5.7 days, though the distribution was skewed due to a few extreme values.  
- Mean final grade (G3) was ~11.9 out of 20, with a standard deviation of 3.2.  

### Categorical variables
- Slightly more female students than male students were included.  
- Most students lived in urban areas and reported internet access at home.  
- A large majority expressed the desire to pursue higher education.  

### Visualizations
- Histograms of grades (G1, G2, G3) showed consistent patterns across terms, with distributions clustered in the mid-range.  
- Boxplots revealed outliers in absences and alcohol consumption (Dalc, Walc).  
- A scatterplot of G1 versus G3 demonstrated a strong positive linear relationship, suggesting that early grades are predictive of final performance.  
- A correlation heatmap was created to show feature relationships across variables.  

### Correlations
- The correlation heatmap confirmed strong positive relationships between G1, G2, and G3.  
- Moderate positive correlations were also observed between parental education (Medu, Fedu) and student grades, suggesting that family background contributes to academic outcomes.  

**Summary:**  
EDA suggests that **grades, absences, and family background** are likely to be important predictors of student performance. These findings will guide the next stage of model selection.

## Model Selection
*To be completed in Week 5.*

## Model Analysis
*To be completed in Week 6.*

## Conclusion & Recommendations
*To be completed after model evaluation.*
