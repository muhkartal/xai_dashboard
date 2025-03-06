"""
predict.py

Utilities for loading a trained model from disk and generating predictions.
Includes:
- load_trained_model (joblib)
- predict_instance
- predict_proba (optional for classification probabilities)

Author: <Muhkartal / kartal.dev>
"""

import os
import sys
import logging
from typing import Dict, List, Union
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


def load_trained_model(model_path: str):
    """
    Loads a trained model from disk (joblib file).
    Exits if the file is not found or fails to load.

    :param model_path: Path to .pkl or .joblib file.
    :return: The trained model object.
    """
    if not os.path.exists(model_path):
        logger.error(f"[FATAL] Model file not found: {model_path}")
        sys.exit(1)

    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded trained model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Error: {e}")
        sys.exit(1)


def predict_instance(model, features: Dict[str, Union[float, int]]) -> int:
    """
    Predicts the class (numeric label) for a single instance of features.
    This is especially relevant if your model expects numeric-coded labels
    (like 0,1,2 for Iris).

    :param model: A trained classification model with .predict()
    :param features: A dictionary of {column_name: value} for one instance
    :return: The predicted class label (integer)
    """
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    return prediction[0]


def predict_proba(model, features: Dict[str, Union[float, int]]) -> float:
    """
    Predicts the probability (max class) for a single instance.

    :param model: A trained classification model with .predict_proba()
    :param features: A dictionary {column_name: value}
    :return: The probability for the predicted class
    """
    df = pd.DataFrame([features])
    proba = model.predict_proba(df).max(axis=1)[0]
    return proba


def predict_batch(model, data: pd.DataFrame):
    """
    Predicts classes for an entire batch of data (DataFrame).
    Useful if you want to pass multiple rows at once.

    :param model: A trained classification model.
    :param data: DataFrame with the same columns as training
    :return: Array of predicted class labels
    """
    predictions = model.predict(data)
    return predictions
