import pandas as pd
import joblib
import os
from load_data import load_data

# -------- Step 1: Load the saved model --------
model_path = "artifacts/diabetes_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Train the model first using train.py")

model = joblib.load(model_path)

# -------- Step 2: Load new input data --------
new_data_path = "data/new_data.csv"

df_new = load_data(new_data_path)

# -------- Step 3: Predict --------
predictions = model.predict(df_new)

# -------- Step 4: Print results --------
print("Predictions for new data:")
print(predictions)
