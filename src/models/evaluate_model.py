import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Load model and test data
model = joblib.load("models/trained_model.pkl")
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")["silica_concentrate"]

# Make predictions
predictions = model.predict(X_test)

# Save predictions
pd.DataFrame({"prediction": predictions}).to_csv("data/processed_data/predictions.csv",index=False)

# Calculate scores
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Save scores
scores = {"mse": mse, "r2": r2}

with open("metrics/scores.json", "w") as file:
    json.dump(scores, file, indent=4)

print("Evaluation completed")
print(scores)
