"""
best_iris_model.py

- Loads iris.csv (with 'variety' as the target, which is string-labeled)
- Encodes 'variety' into numeric labels (0,1,2) so XGBoost won't complain
- Optionally performs minimal preprocessing
- Trains multiple models (RandomForest & XGBoost) with hyperparameter tuning
- Selects the best model based on cross-validated accuracy
- Evaluates final performance on a hold-out test
- Saves the best model to disk

No command-line arguments: everything is in the CONFIG dict below.

Author: <Muhkartal / kartal.dev>

"""

import os
import sys
import logging
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np

# scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# XGBoost
from xgboost import XGBClassifier

import joblib

# Random distributions for hyperparameter search
from scipy.stats import randint, uniform

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
# CONFIG
# -----------------------------------------------------------------------------
CONFIG = {
    # CSV file path (adjust as needed)
    "DATA_PATH": "Github/xai_dashboard/data/iris.csv",

    # The target column in iris.csv (containing e.g. 'Setosa','Versicolor','Virginica')
    "TARGET_COL": "variety",

    # Test size for hold-out
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,

    # Models you want to try
    "MODELS": ["random_forest", "xgboost"],

    # How many cross-validation folds to use for hyperparameter tuning
    "CV": 3,
    # Number of parameter samples in RandomizedSearch
    "N_ITER": 15,

    # Final output model path
    "OUTPUT_PATH": "Github/xai_dashboard/models/best_iris_model.pkl"
}


# -----------------------------------------------------------------------------
# Load & Prepare Data
# -----------------------------------------------------------------------------
def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Loads iris.csv into a pandas DataFrame. Expects 'variety' as the target column.
    """
    if not os.path.exists(data_path):
        logger.error(f"[FATAL] CSV not found at: {data_path}")
        sys.exit(1)
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed reading CSV from {data_path}: {e}")
        sys.exit(1)

def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Encodes string labels in the target column (e.g. 'Setosa','Versicolor','Virginica')
    into numeric labels (0,1,2) so that XGBoost won't raise an error.
    Modifies df in-place and returns df.
    """
    if target_col not in df.columns:
        logger.error(f"[FATAL] Target column '{target_col}' not found in DataFrame.")
        logger.error(f"Columns are: {df.columns.tolist()}")
        sys.exit(1)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    logger.info(f"LabelEncoder mapped: {list(le.classes_)} -> {list(range(len(le.classes_)))}")
    return df

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the DataFrame into train and test sets based on CONFIG.
    Returns: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        f"Data split -> Train size={len(X_train)}, Test size={len(X_test)}. "
        f"Test ratio={test_size}"
    )
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# Model Builders
# -----------------------------------------------------------------------------
def build_random_forest() -> RandomForestClassifier:
    """
    Instantiates a RandomForestClassifier with default parameters.
    """
    return RandomForestClassifier(random_state=CONFIG["RANDOM_STATE"])

def build_xgboost() -> XGBClassifier:
    """
    Instantiates an XGBClassifier with default parameters,
    ensuring no label-encoding conflict with older XGB versions.
    """
    return XGBClassifier(
        random_state=CONFIG["RANDOM_STATE"],
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

def get_model(model_type: str):
    """
    Returns an untrained model instance based on model_type.
    """
    if model_type == "random_forest":
        return build_random_forest()
    elif model_type == "xgboost":
        return build_xgboost()
    else:
        logger.error(f"[FATAL] Model type '{model_type}' is not supported.")
        sys.exit(1)


# -----------------------------------------------------------------------------
# Hyperparameter Distributions
# -----------------------------------------------------------------------------
def get_param_distributions(model_type: str) -> Dict[str, Any]:
    """
    Defines the search space for RandomizedSearchCV, depending on the model.
    """
    if model_type == "random_forest":
        # Example distribution for a random forest
        return {
            "n_estimators": randint(50, 300),
            "max_depth": randint(2, 20),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }
    elif model_type == "xgboost":
        # Example distribution for XGBoost
        return {
            "n_estimators": randint(50, 300),
            "max_depth": randint(2, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.5, 0.5),    # from 0.5 to 1.0
            "colsample_bytree": uniform(0.5, 0.5),  # from 0.5 to 1.0
        }
    else:
        logger.error(f"[FATAL] No hyperparameter distribution for '{model_type}'.")
        sys.exit(1)


# -----------------------------------------------------------------------------
# Model Training with Hyperparameter Tuning
# -----------------------------------------------------------------------------
def train_and_tune_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
):
    """
    1) Build an untrained model
    2) Create a param distribution
    3) RandomizedSearchCV to find best hyperparams
    4) Return the best estimator + the best CV accuracy
    """
    logger.info(f"--- Tuning hyperparams for {model_type} ---")

    base_model = get_model(model_type)
    param_dist = get_param_distributions(model_type)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=CONFIG["N_ITER"],
        cv=CONFIG["CV"],
        random_state=CONFIG["RANDOM_STATE"],
        scoring="accuracy",
        verbose=1,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    logger.info(f"Best params for {model_type}: {random_search.best_params_}")
    logger.info(f"Best CV accuracy for {model_type}: {random_search.best_score_:.4f}")
    return random_search.best_estimator_, random_search.best_score_


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate a trained model on the test set, returning accuracy & F1.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info(f"Test Performance -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    logger.info("Starting best_iris_model pipeline...")

    # 1) Load data
    df = load_dataset(CONFIG["DATA_PATH"])

    # 2) Encode the 'variety' column to numeric labels for XGBoost
    df = encode_target(df, CONFIG["TARGET_COL"])

    # 3) Split data
    X_train, X_test, y_train, y_test = split_data(
        df=df,
        target_col=CONFIG["TARGET_COL"],
        test_size=CONFIG["TEST_SIZE"],
        random_state=CONFIG["RANDOM_STATE"]
    )

    # 4) Train & tune each model, track which is best
    best_model = None
    best_model_name = None
    best_cv_score = -np.inf

    for mtype in CONFIG["MODELS"]:
        tuned_estimator, cv_score = train_and_tune_model(mtype, X_train, y_train)
        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_model_name = mtype
            best_model = tuned_estimator

    logger.info(f"\n+++ BEST MODEL: {best_model_name} with CV accuracy={best_cv_score:.4f} +++")

    # 5) Evaluate best model on test set
    acc, f1 = evaluate(best_model, X_test, y_test)

    # 6) Save best model
    os.makedirs(os.path.dirname(CONFIG["OUTPUT_PATH"]), exist_ok=True)
    joblib.dump(best_model, CONFIG["OUTPUT_PATH"])
    logger.info(f"Best model saved to {CONFIG['OUTPUT_PATH']}")



if __name__ == "__main__":
    main()
