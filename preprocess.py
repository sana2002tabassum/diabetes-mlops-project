# preprocess.py
import pandas as pd
import os

input_path = "data/diabetes.csv"
output_path = "data/diabetes_processed.csv"

df = pd.read_csv(input_path)

# Example preprocessing: fill missing values, drop duplicates
df = df.drop_duplicates()
df = df.fillna(df.mean())

df.to_csv(output_path, index=False)
print(f"Preprocessed data saved to {output_path}")
