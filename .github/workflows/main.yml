import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Folder convention: keep raw files under ./data/raw
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read the file. Load cleaned data CSV (UCI files use semicolon separator)
try:
    df = pd.read_csv(DATA_DIR / "student_clean.csv", sep=";")
    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
except FileNotFoundError:
    print("Error: 'student_clean.csv' not found. Please ensure the file exists in the correct directory.")
    exit()

# Identify categorical columns (object type columns)
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nCategorical columns found: {categorical_cols}")

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Numerical columns: {numerical_cols}")

print("\n" + "="*80)
print("METHOD 1: ONE-HOT ENCODING")
print("="*80)
# One-Hot Encoding: Creates binary columns for each category
# Best for nominal categorical variables (no inherent order)

# Using pandas get_dummies (simplest approach)
df_onehot = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
print(f"\nOne-Hot Encoded Shape: {df_onehot.shape}")
print(f"Column count increased from {len(df.columns)} to {len(df_onehot.columns)}")
print(f"\nFirst few rows of encoded data:\n{df_onehot.head()}")

# Save one-hot encoded data
df_onehot.to_csv(DATA_DIR / 'student_onehot_encoded.csv', index=False)
print(f"\nOne-hot encoded data saved to: {DATA_DIR / 'student_onehot_encoded.csv'}")

print("\n" + "="*80)
print("METHOD 2: LABEL ENCODING")
print("="*80)
# Label Encoding: Converts each category to a number (0, 1, 2, ...)
# Best for ordinal data OR when you want to reduce dimensionality
# WARNING: Can imply false ordering for nominal variables

df_label = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_label[col] = le.fit_transform(df_label[col])
    label_encoders[col] = le
    print(f"\n{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nLabel Encoded Shape: {df_label.shape}")
print(f"\nFirst few rows of label encoded data:\n{df_label.head()}")

# Save label encoded data
df_label.to_csv(DATA_DIR / 'student_label_encoded.csv', index=False)
print(f"\nLabel encoded data saved to: {DATA_DIR / 'student_label_encoded.csv'}")

print("\n" + "="*80)
print("METHOD 3: ORDINAL ENCODING (with custom ordering)")
print("="*80)
# Ordinal Encoding: Like label encoding but you specify the order
# Best for ordinal categorical variables with meaningful order

df_ordinal = df.copy()

# Define ordinal columns and their order
# Based on your data description, these columns have inherent order:
ordinal_mappings = {
    'Medu': [0, 1, 2, 3, 4],  # Education levels already numeric
    'Fedu': [0, 1, 2, 3, 4],  # Education levels already numeric
    # For yes/no columns, we can map them
    'schoolsup': ['no', 'yes'],
    'famsup': ['no', 'yes'],
    'paid': ['no', 'yes'],
    'activities': ['no', 'yes'],
    'nursery': ['no', 'yes'],
    'higher': ['no', 'yes'],
    'internet': ['no', 'yes'],
    'romantic': ['no', 'yes']
}

# Apply ordinal encoding to columns that make sense
for col, order in ordinal_mappings.items():
    if col in df_ordinal.columns and col in categorical_cols:
        oe = OrdinalEncoder(categories=[order])
        df_ordinal[col] = oe.fit_transform(df_ordinal[[col]])
        print(f"\n{col}: {order} -> {list(range(len(order)))}")

# For remaining categorical columns, use label encoding
remaining_categorical = [col for col in categorical_cols if col not in ordinal_mappings.keys()]
for col in remaining_categorical:
    le = LabelEncoder()
    df_ordinal[col] = le.fit_transform(df_ordinal[col])
    print(f"\n{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print(f"\nOrdinal Encoded Shape: {df_ordinal.shape}")
print(f"\nFirst few rows of ordinal encoded data:\n{df_ordinal.head()}")

# Save ordinal encoded data
df_ordinal.to_csv(DATA_DIR / 'student_ordinal_encoded.csv', index=False)
print(f"\nOrdinal encoded data saved to: {DATA_DIR / 'student_ordinal_encoded.csv'}")

print("\n" + "="*80)
print("METHOD 4: MIXED ENCODING using ColumnTransformer")
print("="*80)
# Using ColumnTransformer for applying different encodings to different columns
# This is useful when you want to apply different strategies to different columns

# Define which columns get which encoding
binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 
               'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']

# Filter to only include columns that exist in the dataframe
binary_cols = [col for col in binary_cols if col in categorical_cols]
nominal_cols = [col for col in nominal_cols if col in categorical_cols]

ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse_output=False, drop='first'), nominal_cols),  # One-hot for nominal
        ('label', OrdinalEncoder(), binary_cols)  # Ordinal for binary (saves space)
    ],
    remainder='passthrough'  # Keep numerical columns as they are
)

# Fit and transform
transformed_array = ct.fit_transform(df)

# Get feature names for the transformed data
feature_names = []
# One-hot encoded columns
onehot_features = ct.named_transformers_['onehot'].get_feature_names_out(nominal_cols)
feature_names.extend(onehot_features)
# Binary/ordinal columns
feature_names.extend(binary_cols)
# Remaining numerical columns
feature_names.extend(numerical_cols)

# Convert back to DataFrame
df_mixed = pd.DataFrame(transformed_array, columns=feature_names)
print(f"\nMixed Encoding Shape: {df_mixed.shape}")
print(f"\nFirst few rows of mixed encoded data:\n{df_mixed.head()}")

# Save mixed encoded data
df_mixed.to_csv(DATA_DIR / 'student_mixed_encoded.csv', index=False)
print(f"\nMixed encoded data saved to: {DATA_DIR / 'student_mixed_encoded.csv'}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Original data shape: {df.shape}")
print(f"One-Hot Encoded: {df_onehot.shape} - Best for nominal categories, but creates many columns")
print(f"Label Encoded: {df_label.shape} - Compact, but implies false ordering")
print(f"Ordinal Encoded: {df_ordinal.shape} - Good for ordered categories")
print(f"Mixed Encoded: {df_mixed.shape} - Balanced approach using different methods")
print("\nAll encoded datasets have been saved to the data directory!")
print("\nRecommendation: For machine learning with this student dataset,")
print("use One-Hot encoding or Mixed encoding approach for best results.")
