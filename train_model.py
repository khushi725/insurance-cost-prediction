import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("insurance.csv")

# Convert smoker column to numeric
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

# Select features
X = data[["age", "bmi", "smoker", "gender", "children", "income"]]
y = data['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")
print("Model Accuracy (R^2 score):", model.score(X_test, y_test))