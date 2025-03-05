"""
train_model.py (No argparse version)

A streamlined script for training a classification model (e.g., RandomForest)
with optional hyperparameter tuning, but without any command-line argument parsing.

Main Steps:
1. Data Loading
2. Data Splitting
3. Model Initialization
4. (Optional) Hyperparameter Tuning
5. Model Training & Evaluation
6. Model Saving

Configure the script by editing the CONFIG dict below or
using environment variables (recommended).

Example Usage:
--------------
Simply run:
    python -m src.train_model

Author: <Your Name>
"""

import os
import sys
import logging
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import randint, uniform
import joblib

# -----------------------------------------------------------------------------
# Configure Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
CONFIG = {
    # Data
    "DATA_PATH": os.getenv("DATA_PATH", "Github/xai_dashboard/data/iris.csv"),
    "TARGET_COL": os.getenv("TARGET_COL", "variety"),
    # Train/Test Split
    "TEST_SIZE": float(os.getenv("TEST_SIZE", 0.2)),
    "RANDOM_STATE": int(os.getenv("RANDOM_STATE", 42)),
    # Model
    "MODEL_TYPE": os.getenv("MODEL_TYPE", "random_forest"),
    # Hyperparameter Tuning
    "ENABLE_HP_TUNING": bool(int(os.getenv("ENABLE_HP_TUNING", "0"))),  # 0 or 1
    "CV": int(os.getenv("CV", 3)),
    "HP_N_ITER": int(os.getenv("HP_N_ITER", 10)),
    # Output
    "OUTPUT_PATH": os.getenv("OUTPUT_PATH", "/models/model.pkl"),
}


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file into a pandas DataFrame.

    :param data_path: Path to the CSV file containing features and target.
    :return: DataFrame with loaded data.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset from {data_path}. Error: {e}")
        sys.exit(1)


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits DataFrame into training and testing sets.

    :param df: Full DataFrame (features + target column).
    :param target_col: Name of the target column.
    :param test_size: Fraction of data for test set.
    :param random_state: Random seed for reproducibility.
    :return: (X_train, X_test, y_train, y_test)
    """
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame.")
        sys.exit(1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        f"Data split -> train_size={len(X_train)}, test_size={len(X_test)}. "
        f"Test Ratio={test_size}"
    )
    return X_train, X_test, y_train, y_test


def get_model(model_type: str = "random_forest") -> object:
    """
    Returns an initialized model based on the given type.

    :param model_type: Type of model. Currently supports "random_forest".
    :return: The instantiated model object.
    """
    if model_type.lower() == "random_forest":
        model = RandomForestClassifier()
        return model
    else:
        logger.error(f"Model type '{model_type}' is not supported.")
        sys.exit(1)


def hyperparameter_tuning(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3,
    n_iter: int = 10,
    random_state: int = 42
) -> object:
    """
    Uses RandomizedSearchCV to find better hyperparameters.

    :param model: The base model to tune (e.g., RandomForestClassifier).
    :param X_train: Training features.
    :param y_train: Training labels.
    :param cv: Cross-validation folds.
    :param n_iter: Number of sampled parameter settings.
    :param random_state: Seed for reproducibility.
    :return: Best estimator from the search.
    """
    # Example distributions for a random forest:
    param_distributions = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(2, 20),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    logger.info(f"Best Hyperparameters: {random_search.best_params_}")
    logger.info(f"Best CV Score: {random_search.best_score_:.4f}")

    return random_search.best_estimator_


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates the model on test data using accuracy and weighted F1.

    :param model: Trained model object with a .predict() method.
    :param X_test: Test feature matrix.
    :param y_test: True labels for test.
    :return: Dict containing accuracy and F1 score.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    metrics = {"accuracy": acc, "f1_score": f1}
    logger.info(f"Evaluation -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return metrics


# Scikit-learn metrics needed in evaluate_model
from sklearn.metrics import accuracy_score, f1_score


def main(config: dict):
    """
    Main training flow, orchestrated by the config dictionary:
    1) Load dataset from CSV
    2) Split into train/test
    3) Initialize model
    4) (Optional) hyperparameter tuning
    5) Train final model
    6) Evaluate model on test
    7) Save model artifact
    """
    logger.info("Starting training...")

    # 1) Load data
    df = load_dataset(config["DATA_PATH"])

    # 2) Split data
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_col=config["TARGET_COL"],
        test_size=config["TEST_SIZE"],
        random_state=config["RANDOM_STATE"]
    )

    # 3) Get model
    model = get_model(model_type=config["MODEL_TYPE"])

    # 4) Hyperparameter tuning (optional)
    if config["ENABLE_HP_TUNING"]:
        model = hyperparameter_tuning(
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv=config["CV"],
            n_iter=config["HP_N_ITER"],
            random_state=config["RANDOM_STATE"]
        )
    else:
        logger.info("Training model with default hyperparameters...")
        model.fit(X_train, y_train)

    # 5) Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # 6) Save model
    os.makedirs(os.path.dirname(config["OUTPUT_PATH"]), exist_ok=True)
    joblib.dump(model, config["OUTPUT_PATH"])
    logger.info(f"Model saved to {config['OUTPUT_PATH']}")
    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    # Instead of argparse, we simply define or use the CONFIG dictionary directly.
    # Optionally, you can edit CONFIG or set environment variables before running.
    main(CONFIG)
