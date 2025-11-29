import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate():
    # Load processed data
    df = pd.read_csv("data/diabetes_processed.csv")
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Load trained model
    model = joblib.load("model/diabetes_model.pkl")

    # Predictions
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }

    # Save metrics to file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation complete. Metrics saved to metrics.json")


if __name__ == "__main__":
    evaluate()
