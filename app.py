from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["cp"])
    bp = float(request.form["bp"])
    chol = float(request.form["chol"])
    fbs = float(request.form["fbs"])
    restecg = float(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = float(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["slope"])
    ca = float(request.form["ca"])
    thal = float(request.form["thal"])

    features = np.array([[age, sex, cp, bp, chol, fbs,
                          restecg, thalach, exang,
                          oldpeak, slope, ca, thal]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)

    # 🔍 DEBUG OUTPUT (CHECK TERMINAL)
    print("Raw Input:", features)
    print("Scaled Input:", features_scaled)
    print("Prediction:", prediction)
    print("Probability:", probability)

    if prediction == 1:
        result = "⚠️ Heart Disease Detected"
    else:
        result = "✅ No Heart Disease"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
