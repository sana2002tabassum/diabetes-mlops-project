# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Paths
data_path = "data/diabetes_processed.csv"
model_dir = "model"
model_path = os.path.join(model_dir, "diabetes_model.pkl")

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

# Load processed data
df = pd.read_csv(data_path)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
