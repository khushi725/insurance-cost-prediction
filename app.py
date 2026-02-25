from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("insurance_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])
        region = int(request.form["region"])

        data = np.array([[age, sex, bmi, children, smoker, region]])

        prediction = model.predict(data)

        output = round(prediction[0], 2)

        return render_template("index.html",
               prediction_text=f"Estimated Insurance Cost: ₹ {output}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)