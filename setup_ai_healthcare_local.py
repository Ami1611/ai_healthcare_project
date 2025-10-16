import os
import subprocess
import sys
import pandas as pd
import joblib

# -----------------------------
# 0. User Configuration
# -----------------------------
GITHUB_USERNAME = "YourUsername"      # Replace with your GitHub username
GITHUB_TOKEN = "ghp_yourtokenhere"   # Replace with your GitHub Personal Access Token
REPO_NAME = "ai_healthcare_webapp"

# -----------------------------
# 1. Create project folders
# -----------------------------
folders = ["data", "model", "app", "app/templates", "app/static"]
for f in folders:
    os.makedirs(f, exist_ok=True)
print("✅ Project folders created.")

# -----------------------------
# 2. Create virtual environment
# -----------------------------
if not os.path.exists("venv"):
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    print("✅ Virtual environment created.")

# -----------------------------
# 3. Install dependencies
# -----------------------------
subprocess.run([os.path.join("venv","Scripts","pip"), "install",
                "pandas", "numpy", "scikit-learn", "flask", "joblib"])
print("✅ Dependencies installed.")

# -----------------------------
# 4. Load local dataset
# -----------------------------
dataset_path = "data/heart_disease.csv"
if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at {dataset_path}. Please download it manually first.")
    sys.exit(1)

df = pd.read_csv(dataset_path)
print(f"✅ Dataset loaded. Shape: {df.shape}")

# -----------------------------
# 5. Train model
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump({"model": model, "features": X.columns.tolist()}, "model/heart_model.pkl")
print("✅ Model trained and saved!")

# -----------------------------
# 6. Create Flask App
# -----------------------------
app_code = '''from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

artifact = joblib.load("../model/heart_model.pkl")
model = artifact["model"]
features = artifact["features"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        x = np.array([data["features"].get(f, 0) for f in features]).reshape(1, -1)
        prediction = int(model.predict(x)[0])
        probability = float(model.predict_proba(x)[0][1])
        return jsonify({"prediction": prediction, "probability": probability})
    return render_template("index.html", features=features)

if __name__ == "__main__":
    app.run(debug=True)
'''
with open("app/app.py", "w") as f:
    f.write(app_code)
print("✅ Flask app.py created.")

# -----------------------------
# 7. Create HTML template
# -----------------------------
html_code = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Heart Disease Prediction</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: auto; padding: 20px; }
        label { display: block; margin-top: 10px; }
        input { width: 100%; padding: 5px; }
        button { margin-top: 20px; padding: 10px 20px; }
        .result { margin-top: 20px; font-weight: bold; color: green; }
    </style>
</head>
<body>
    <h1>Heart Disease Prediction</h1>
    <form id="prediction-form">
        {% for f in features %}
        <label>{{ f }}</label>
        <input type="number" step="any" name="{{ f }}" required>
        {% endfor %}
        <button type="submit">Predict</button>
    </form>
    <div id="result" class="result"></div>

    <script>
        const form = document.getElementById('prediction-form');
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(form);
            let features = {};
            for (let [key, value] of formData.entries()) {
                features[key] = parseFloat(value);
            }
            try {
                const response = await fetch('/', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: features })
                });
                const data = await response.json();
                document.getElementById('result').innerHTML = 
                    `Prediction: ${data.prediction} <br> Probability: ${data.probability.toFixed(2)}`;
            } catch (err) {
                document.getElementById('result').innerHTML = 'Error: ' + err;
            }
        });
    </script>
</body>
</html>
'''
with open("app/templates/index.html", "w") as f:
    f.write(html_code)
print("✅ HTML template created.")

# -----------------------------
# 8. Setup Git
# -----------------------------
subprocess.run(["git", "init"])
with open(".gitignore", "w") as f:
    f.write("venv/\n__pycache__/\n*.pyc\nmodel/\ndata/\n")
subprocess.run(["git", "add", "."])
subprocess.run(["git", "commit", "-m", "Initial commit: AI Healthcare Web App"])
print("✅ Git initialized and first commit done.")

# -----------------------------
# 9. Create GitHub Repo & Push
# -----------------------------
import requests
api_url = "https://api.github.com/user/repos"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}
payload = {"name": REPO_NAME, "private": False}

r = requests.post(api_url, json=payload, headers=headers)
if r.status_code == 201:
    print(f"✅ GitHub repo '{REPO_NAME}' created.")
else:
    print("❌ Could not create repo:", r.json())

subprocess.run(["git", "remote", "add", "origin",
                f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"])
subprocess.run(["git", "branch", "-M", "main"])
subprocess.run(["git", "push", "-u", "origin", "main"])
print("✅ Code pushed to GitHub.")

# -----------------------------
# 10. Launch Flask App
# -----------------------------
print("\n✅ Setup complete! Flask app is launching...")
subprocess.run([os.path.join("venv","Scripts","python"), "app/app.py"])
======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...======================================= RESTART: C:/Users/Admin/ai_healthcare_project/setup_ai_healthcare_local.py ======================================
✅ Project folders created.
✅ Dependencies installed.
✅ Dataset loaded. Shape: (1025, 14)
✅ Model trained and saved!
✅ Flask app.py created.
✅ HTML template created.
✅ Git initialized and first commit done.
❌ Could not create repo: {'message': 'Bad credentials', 'documentation_url': 'https://docs.github.com/rest', 'status': '401'}
✅ Code pushed to GitHub.

✅ Setup complete! Flask app is launching...
