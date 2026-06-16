import pandas as pd
from sklearn.model_selection import train_test_split

# Read the raw data
df = pd.read_csv("data/raw_data/raw.csv")

# Separate features X and target y
X = df.drop(["date", "silica_concentrate"], axis=1)
y = df["silica_concentrate"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Save the four output files
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)

print("Data split completed")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
