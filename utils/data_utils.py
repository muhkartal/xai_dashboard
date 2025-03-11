import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, Optional, List, Union, Any

logger = logging.getLogger(__name__)

def load_user_csv(uploaded_csv) -> Optional[pd.DataFrame]:
    """
    Reads a user-uploaded CSV file into a pandas DataFrame with improved error handling.
    """
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            logger.info(f"User CSV loaded successfully, shape={df.shape}.")
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return None
    return None

def get_feature_info(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Extract metadata about each feature in the dataframe.
    Useful for creating dynamic UI controls and understanding data distribution.

    Returns:
        Dictionary with feature names as keys, each containing:
        - data type
        - min/max/mean/median for numeric features
        - unique values for categorical features (if not too many)
        - missing value count
        - etc.
    """
    feature_info = {}

    for col in df.columns:
        col_type = df[col].dtype
        info = {'type': str(col_type)}

        # Count missing values
        missing_count = df[col].isna().sum()
        info['missing_count'] = int(missing_count)
        info['missing_percent'] = float(missing_count / len(df) * 100)

        # For numeric columns
        if np.issubdtype(col_type, np.number):
            info['min'] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            info['max'] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            info['mean'] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
            info['median'] = float(df[col].median()) if not pd.isna(df[col].median()) else None
            info['std'] = float(df[col].std()) if not pd.isna(df[col].std()) else None

        # For categorical/string columns
        else:
            unique_vals = df[col].unique()
            info['unique_count'] = len(unique_vals)
            if len(unique_vals) <= 10:  # Only store if not too many
                info['unique_values'] = [str(val) for val in unique_vals]
                # Calculate frequency of each value
                value_counts = df[col].value_counts().to_dict()
                info['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}

        feature_info[col] = info

    return feature_info

def plot_feature_distributions(df: pd.DataFrame, feature_info: Dict, max_plots: int = 10):
    """
    Create visualizations for feature distributions.
    For numerical features: histograms
    For categorical features: bar charts

    Returns a dict of matplotlib figures keyed by feature name.
    """
    plots = {}
    count = 0

    # Sort features with numerical first, then categorical
    numerical_features = [col for col in df.columns
                         if np.issubdtype(df[col].dtype, np.number)]
    categorical_features = [col for col in df.columns
                           if col not in numerical_features]

    # Process numerical features
    for col in numerical_features:
        if count >= max_plots:
            break

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        plots[col] = fig
        count += 1

    # Process categorical features
    for col in categorical_features:
        if count >= max_plots:
            break

        # Only plot if not too many unique values
        if feature_info[col]['unique_count'] <= 10:
            fig, ax = plt.subplots(figsize=(8, 4))
            value_counts = df[col].value_counts().sort_values(ascending=False).head(10)
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title(f"Distribution of {col}")
            plots[col] = fig
            count += 1

    return plots

def prepare_sample_instance(df: pd.DataFrame, feature_info: Dict) -> Dict:
    """
    Generate a representative sample instance from the dataset.
    Useful as a starting point for the user to modify.
    """
    sample = {}

    # For each feature, use the median for numeric, most common value for categorical
    for col, info in feature_info.items():
        if 'median' in info and info['median'] is not None:
            # Use median for numeric
            sample[col] = info['median']
        elif 'value_counts' in info and len(info['value_counts']) > 0:
            # Use most common value for categorical
            most_common = max(info['value_counts'].items(), key=lambda x: x[1])[0]
            sample[col] = most_common
        else:
            # Fallback
            sample[col] = df[col].iloc[0] if len(df) > 0 else None

    return sample

@st.cache_data
def get_correlation_matrix(df: pd.DataFrame, method: str = 'pearson'):
    """
    Calculate correlation matrix for numerical features with caching.
    """
    numerics = df.select_dtypes(include=[np.number])
    if numerics.shape[1] < 2:  # Need at least 2 numeric columns
        return None

    return numerics.corr(method=method)
