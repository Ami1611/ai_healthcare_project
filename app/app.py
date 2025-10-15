from flask import Flask, request, render_template, jsonify
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
