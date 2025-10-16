from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("model/heart_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        # Prepare input
        x = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                       thalach, exang, oldpeak, slope, ca, thal]])

        # Predict
        prediction = model.predict(x)[0]
        probability = model.predict_proba(x)[0][prediction] * 100

        # Results
        if prediction == 1:
            result = "⚠️ Potential signs of heart disease detected."
            explanation = "Please consult a cardiologist for a detailed medical check-up."
        else:
            result = "✅ Your heart health appears normal."
            explanation = "Keep maintaining a healthy lifestyle and regular check-ups."

        return render_template(
            'index.html',
            prediction_text=result,
            confidence=f"Model Confidence: {probability:.2f}%",
            advice=explanation
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
