"""
data_preprocessing.py

Utilities for loading and preprocessing a dataset prior to model training.
Includes:
- load_data
- encode_target (optional for label-encoding your target)
- drop_duplicates / fill_missing (examples)
- train_test_split wrapper if desired

Author: <Muhkartal / kartal.dev>
"""

import os
import sys
import logging
from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.
    Exits if the file is not found or fails to load.

    :param data_path: Path to the CSV file.
    :return: DataFrame with the loaded data.
    """
    if not os.path.exists(data_path):
        logger.error(f"[FATAL] File not found: {data_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape={df.shape} from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read CSV from {data_path}: {e}")
        sys.exit(1)


def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Encodes a string-based target column (like 'variety') into numeric labels
    using LabelEncoder. This is critical for certain models (e.g., XGBoost).

    :param df: DataFrame containing the target column.
    :param target_col: Name of the target column (e.g. 'variety').
    :return: DataFrame with the target column replaced by encoded numeric labels.
    """
    if target_col not in df.columns:
        logger.error(f"[FATAL] Target column '{target_col}' not found in DataFrame.")
        logger.error(f"Columns are: {df.columns.tolist()}")
        sys.exit(1)

    le = LabelEncoder()
    original_classes = df[target_col].unique()
    df[target_col] = le.fit_transform(df[target_col])
    logger.info(f"LabelEncoder mapped {list(original_classes)} to {list(le.classes_)} (0..N).")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops duplicate rows if any. Useful for data cleaning.

    :param df: Input DataFrame.
    :return: DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} duplicate rows.")
    return df


def fill_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Example function for filling missing numeric values. Not strictly needed
    for the Iris dataset, but helpful for real-world usage.

    :param df: Input DataFrame.
    :param strategy: "mean" or "median" or "zero".
    :return: DataFrame with missing values in numeric columns filled.
    """
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if strategy not in ["mean", "median", "zero"]:
        logger.warning(f"Unknown strategy '{strategy}'. No filling performed.")
        return df

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == "mean":
                fill_val = df[col].mean()
            elif strategy == "median":
                fill_val = df[col].median()
            else:  # zero
                fill_val = 0
            df[col].fillna(fill_val, inplace=True)
    logger.info(f"Filled missing values with strategy='{strategy}' for numeric columns.")
    return df
