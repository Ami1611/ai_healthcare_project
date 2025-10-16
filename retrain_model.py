# app/retrain_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# === Step 1: Load latest data ===
data_path = "data/latest_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data file not found at {data_path}")

print("📊 Loading data...")
data = pd.read_csv(data_path)

# === Step 2: Preprocess data ===
if "target" not in data.columns:
    raise ValueError("❌ 'target' column not found in dataset.")

X = data.drop("target", axis=1)
y = data["target"]

# === Step 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 4: Train model ===
print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 5: Evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Model Accuracy: {accuracy:.4f}")

# === Step 6: Save model if accuracy acceptable ===
model_path = "model/heart_model.pkl"
if accuracy >= 0.80:
    joblib.dump(model, model_path)
    print(f"✅ Model updated and saved to: {model_path}")
else:
    print("⚠️ Model not saved: accuracy below threshold (0.80)")
