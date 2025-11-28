# evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Paths
data_path = "data/diabetes_processed.csv"
model_path = "model/diabetes_model.pkl"

# Load processed data
df = pd.read_csv(data_path)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Load trained model
model = joblib.load(model_path)

# Make predictions
y_pred = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

