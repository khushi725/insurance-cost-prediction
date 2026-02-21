from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    smoker = float(request.form["smoker"])
    gender = float(request.form["gender"])
    children = float(request.form["children"])
    income = float(request.form["income"])

    features = np.array([[age, bmi, smoker, gender, children, income]])

    prediction = model.predict(features)

    return render_template("index.html",
                          prediction_text="Estimated Insurance Cost: ₹ {:.2f}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)