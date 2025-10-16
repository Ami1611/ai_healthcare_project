import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# -----------------------------
# 1. Load Dataset
# -----------------------------
data_path = "data/heart_disease.csv"  # Make sure this exists
df = pd.read_csv(data_path)

# -----------------------------
# 2. Preprocessing
# -----------------------------
# Optional: rename columns for clarity if needed
# df.columns = ['age', 'sex', 'cp', ...]

# Features and target
X = df.drop("target", axis=1)  # "target" is heart disease label
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 5. Save Model
# -----------------------------
# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save trained model directly
joblib.dump(model, "model/heart_model.pkl")
print("✅ Model trained and saved at 'model/heart_model.pkl'")
