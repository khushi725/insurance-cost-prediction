import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("insurance.csv")

# Convert categorical to numeric
data["sex"] = data["sex"].map({"male":1, "female":0})
data["smoker"] = data["smoker"].map({"yes":1, "no":0})

# Convert region using one-hot encoding
data = pd.get_dummies(data, columns=["region"], drop_first=True)

# Features
X = data.drop("charges", axis=1)
y = data["charges"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")