# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import mlflow
import mlflow.sklearn
import logging
import os

# -----------------------
# Setup logging
# -----------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

logging.info("Training script started...")

# -----------------------
# Load processed data
# -----------------------
data_path = "data/diabetes_processed.csv"
df = pd.read_csv(data_path)
logging.info(f"Loaded processed dataset from {data_path}")

# -----------------------
# Split data
# -----------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logging.info("Completed train-test split.")

# -----------------------
# Train model
# -----------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
logging.info("Model training completed.")

# -----------------------
# Save model
# -----------------------
model_output_path = "models/model.pkl"
os.makedirs("models", exist_ok=True)

with open(model_output_path, "wb") as f:
    pickle.dump(model, f)

logging.info(f"Model saved to {model_output_path}")

# -----------------------
# MLflow Logging
# -----------------------
mlflow.set_experiment("Diabetes Prediction Experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_artifact(model_output_path)
    mlflow.log_artifact(os.path.join(LOG_DIR, "train.log"))

logging.info("Training script finished and logged to MLflow.")
