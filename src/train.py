import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.load_data import load_data

# -------- Step 1: Load the dataset --------
data_path = "data/diabetes.csv"
df = load_data(data_path)

# -------- Step 2: Split features and target --------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------- Step 3: Train-test split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------- Step 4: Train model --------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------- Step 5: Evaluate model --------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# -------- Step 6: Create artifacts folder --------
os.makedirs("artifacts", exist_ok=True)

# -------- Step 7: Save model --------
model_path = "artifacts/diabetes_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")
