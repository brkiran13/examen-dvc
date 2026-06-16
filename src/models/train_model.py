import joblib
import pandas as pd
from sklearn.linear_model import Ridge

# Load the normalized training features
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")

# Load the training target
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]

# Load the best parameters found by GridSearch
best_params = joblib.load("models/best_params.pkl")

# Create the Ridge model with the best alpha
model = Ridge(alpha=best_params["alpha"])

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/trained_model.pkl")

print("Model training completed")
print("Trained model saved to models/trained_model.pkl")
