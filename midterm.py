

# Add any other imports you need here, e.g.:
# from sklearn.linear_model import LogisticRegression

# Define necessary directory names directly
DATASETS_DIR = 'datasets'
TEST_DIR = 'test'         # Directory where predictions will be saved
SEED = 42

# --- STUDENT CODE: MODIFY ONLY THE FUNCTION BELOW ---






import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score, make_scorer
)

from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb

def print_data_info(X_train, y_train):
    """Prints essential information about the dataset."""
    
    # Convert to pandas DataFrame for easier inspection
    X_train_df = pd.DataFrame(X_train)
    
    # Number of samples and features
    num_samples, num_features = X_train.shape
    print(f"Number of samples: {num_samples}")
    print(f"Number of features: {num_features}")
    
    # Feature summary statistics (mean, std, min, max)
    print("\nFeature Summary Statistics:")
    print(X_train_df.describe())
    
    # Check for missing values
    missing_values = X_train_df.isnull().sum()
    print("\nMissing values in features:")
    print(missing_values[missing_values > 0])
    
    # Check if target is classification or regression
    unique_classes = len(np.unique(y_train))
    if unique_classes < 20:
        print("\nTarget variable appears to be a classification problem.")
    else:
        print("\nTarget variable appears to be a regression problem.")
    
    # Distribution of the target variable for classification problems
    if unique_classes < 20:
        print("\nClass distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))
    else:
        print("\nTarget variable distribution (for regression):")
        print(f"Mean: {np.mean(y_train):.2f}, Std: {np.std(y_train):.2f}, Min: {np.min(y_train):.2f}, Max: {np.max(y_train):.2f}")
        
    # Correlation matrix for features (useful for regression)
    if unique_classes >= 20:  # only for regression
        corr_matrix = X_train_df.corr()
        print("\nCorrelation matrix for features:")
        print(corr_matrix)
        
    print("\nData Info Summary:")
    print(f"Is classification: {unique_classes < 20}")
    print(f"Feature count: {num_features}")
    print(f"Sample count: {num_samples}")
    print(f"Missing values: {missing_values[missing_values > 0].sum()}")


def is_classification(y_train):
    return len(np.unique(y_train)) <= 20 and np.issubdtype(y_train.dtype, np.integer)

# Helper function to try and test different models
def get_best_model(X_train, y_train, X_val, y_val):
    if is_classification(y_train):
        # Classification models (with reduced search space)
        candidates = [
            {
                "name": "RandomForest",
                "model": RandomForestClassifier(random_state=42),
                "param_distributions": {
                    'n_estimators': [50],  # Only 50 trees for faster training
                    'max_depth': [5],  # Limit tree depth
                }
            },
            {
                "name": "LogisticRegression",
                "model": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
                "param_distributions": {
                    'clf__C': [1],  # Regularization strength
                    'clf__solver': ['liblinear'],
                    'clf__penalty': ['l2']
                }
            },
            {
                "name": "SVC",
                "model": Pipeline([('scaler', StandardScaler()), ('clf', SVC(random_state=42))]),
                "param_distributions": {
                    'clf__C': [1],  # Regularization strength
                    'clf__kernel': ['linear'],
                    'clf__gamma': ['scale']
                }
            },
            {
                "name": "XGBoost",
                "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                "param_distributions": {
                    'n_estimators': [50],  # Limit to 50 estimators
                    'max_depth': [3],  # Restrict tree depth
                    'learning_rate': [0.1],
                    'subsample': [0.8]
                }
            },
            {
                "name": "GradientBoosting",
                "model": GradientBoostingClassifier(random_state=42),
                "param_distributions": {
                    'n_estimators': [50],  # Limit to 50 estimators
                    'max_depth': [3],  # Restrict tree depth
                    'learning_rate': [0.1]
                }
            }
        ]
        scoring = 'accuracy'
    else:
        # Regression models (with reduced search space)
        candidates = [
            {
                "name": "RandomForest",
                "model": RandomForestRegressor(random_state=42),
                "param_distributions": {
                    'n_estimators': [50],  # Only 50 trees for faster training
                    'max_depth': [5],  # Limit tree depth
                }
            },
            {
                "name": "LinearRegression",
                "model": LinearRegression(),
                "param_distributions": {}
            },
            {
                "name": "SVR",
                "model": Pipeline([('scaler', StandardScaler()), ('clf', SVR())]),
                "param_distributions": {
                    'clf__C': [1],
                    'clf__kernel': ['linear']
                }
            },
            {
                "name": "XGBoost",
                "model": XGBRegressor(random_state=42),
                "param_distributions": {
                    'n_estimators': [50],  # Limit to 50 estimators
                    'max_depth': [3],  # Restrict tree depth
                    'learning_rate': [0.1],
                    'subsample': [0.8]
                }
            },
            {
                "name": "GradientBoosting",
                "model": GradientBoostingRegressor(random_state=42),
                "param_distributions": {
                    'n_estimators': [50],  # Limit to 50 estimators
                    'max_depth': [3],  # Restrict tree depth
                    'learning_rate': [0.1]
                }
            }
        ]
        scoring = 'neg_mean_squared_error'

    best_model = None
    best_score = -np.inf if scoring == 'neg_mean_squared_error' else -1
    best_name = ""
    best_params = None

    # Cross-validation setup with fewer folds (n_splits=3)
    cv = StratifiedKFold(n_splits=3) if is_classification(y_train) else KFold(n_splits=3)

    # Try all models
    for candidate in candidates:
        rand_search = RandomizedSearchCV(candidate["model"], candidate["param_distributions"],
                                        n_iter=5, cv=cv, scoring=scoring, n_jobs=-1)  # Reduced n_iter
        rand_search.fit(X_train, y_train)  # Use the full training data for the search

        # Evaluate on validation set
        val_preds = rand_search.predict(X_val)
        score = accuracy_score(y_val, val_preds) if is_classification(y_train) else -mean_squared_error(y_val, val_preds)
        
        print(f"{candidate['name']} validation score: {score:.4f}")

        # Track best model based on validation score
        if score > best_score:
            best_score = score
            best_model = rand_search.best_estimator_
            best_name = candidate["name"]
            best_params = rand_search.best_params_

    print(f"\nSelected Model: {best_name}")
    print("Best hyperparameters:", best_params)
    print("Validation score:", best_score)

    # Retrain best model on the combined training + validation data
    best_model.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))

    return best_model
# Main function for model selection and prediction
def train_predict(X_train, y_train, X_test):
    # Split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Get the best model after testing different models
    best_model = get_best_model(X_train_split, y_train_split, X_val, y_val)

    # Predict on test set
    return best_model.predict(X_test)

if __name__ == '__main__':
    # --- Configuration ---
    # Enter your FULL 9-digit student ID here (e.g., '123456789')
    student_id_full = '213758758'  # CHANGE THIS TO YOUR FULL STUDENT ID ###

    # We use the LAST 5 digits for file naming convention
    student_id = student_id_full[-5:]

    # --- Sanity Check ---
    if len(student_id_full) != 9 or not student_id_full.isdigit():
        print(f"Error: Entered student ID '{student_id_full}' is not a valid 9-digit number.")
        print("Please enter your complete 9-digit ID.")
        exit()
    print(f"Using derived ID for files: {student_id} (from full ID: {student_id_full})")

    # --- Load Data ---
    # Assumes student data file is in the 'datasets' directory relative to this script
    train_file = os.path.join(DATASETS_DIR, f'{student_id}_train.npz')
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        print(f"Contact your TA")
        exit()

    data = np.load(train_file)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']

    print(f'Running for student ID: {student_id}')
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')

    # --- Data Info ---
    print_data_info(X_train, y_train)

    # --- Train and Predict ---
    # Train the model and generate predictions for the test set
    test_predictions = train_predict(X_train, y_train, X_test)

    # --- Save Predictions ---
    # Ensure the output directory exists
    os.makedirs(TEST_DIR, exist_ok=True)

    # Save the test predictions to a file
    pred_file = os.path.join(TEST_DIR, f'{student_id}_test_predictions.npz')
    np.savez(pred_file, test_predictions=test_predictions)

    print(f'\nPredictions saved to {pred_file}')
    print('Submission file created successfully!')
