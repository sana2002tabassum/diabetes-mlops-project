import mlflow
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import pickle

# Load model and data
model = pickle.load(open("models/model.pkl", "rb"))  # updated path
data = pd.read_csv("data/diabetes_processed.csv")  # update path if needed
X = data.drop("Outcome", axis=1)  # check column name
y = data["Outcome"]

# Predict and calculate metrics
y_pred = model.predict(X)
metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "f1_score": f1_score(y, y_pred),
    "precision": precision_score(y, y_pred),
    "recall": recall_score(y, y_pred)
}

# Save metrics locally
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation metrics:", metrics)

# Log metrics to MLflow (optional)
with mlflow.start_run():
    mlflow.log_metrics(metrics)
