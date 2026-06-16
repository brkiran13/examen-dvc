import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Load the normalized training features
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")

# Load the training target
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]

# Ridge regrssion model is used
model = Ridge()

# For Ridge alpha is the value used for Gridsearch
params = {"alpha": [0.1, 1.0, 10.0, 100.0]}

# r2 is used to choose the best parameter from grid search
grid = GridSearchCV(model, params, cv=3, scoring="r2")

# Run the grid search on the training data
grid.fit(X_train, y_train)

# Save the best parameters 
joblib.dump(grid.best_params_, "models/best_params.pkl")

print("GridSearch completed")
print("Best parameters:", grid.best_params_)
