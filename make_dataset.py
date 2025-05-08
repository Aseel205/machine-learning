import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os

# Load Fashion-MNIST dataset
data = fetch_openml('fashion-mnist', version=1)

# Extract features and target labels
X = data.data
y = data.target.astype(int)  # Convert target to integer

# Limit the dataset to 15,000 samples (between 10,000 - 20,000)
X = X[:15000]
y = y[:15000]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure the directory exists
os.makedirs('datasets', exist_ok=True)

# Save dataset in npz format
np.savez('datasets/58758_train.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

print("Dataset saved successfully!")
