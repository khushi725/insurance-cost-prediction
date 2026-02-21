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
    sex = float(request.form["sex"])
    bmi = float(request.form["bmi"])
    children = float(request.form["children"])
    smoker = float(request.form["smoker"])
    region = request.form["region"]

    # Region encoding (must match training)
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    features = np.array([[age, sex, bmi, children, smoker,
                          region_northwest,
                          region_southeast,
                          region_southwest]])

    prediction = model.predict(features)

    return render_template("index.html",
                           prediction_text="Estimated Insurance Cost: ₹ {:.2f}".format(prediction[0]))

if __name__ == "__main__":
    app.run()