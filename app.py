from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    smoker = request.form["smoker"]

    smoker = 1 if smoker == "yes" else 0

    features = np.array([[age, bmi, smoker]])
    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template("index.html",
                           prediction_text=f"Estimated Insurance Cost: ${output}")

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)