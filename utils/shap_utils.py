import io
import tempfile
import os
import base64
import logging
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import streamlit as st

logger = logging.getLogger(__name__)

@st.cache_resource
def init_shap_explainer(model, background_df=None, max_samples=100):
    """
    Initialize SHAP explainer with caching for better performance
    """
    # Sample background data for performance if needed
    if background_df is not None and len(background_df) > max_samples:
        background_df = background_df.sample(max_samples, random_state=42)

    try:
        # Try TreeExplainer first (much faster for tree-based models)
        return shap.TreeExplainer(model, data=background_df)
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}. Trying KernelExplainer...")
        try:
            # Fall back to KernelExplainer for non-tree models
            prediction_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            return shap.KernelExplainer(prediction_fn, background_df)
        except Exception as e2:
            logger.error(f"All SHAP explainers failed: {e2}")
            return None

def generate_shap_explanation(explainer, instance_df, plot_type='bar'):
    """
    Generate SHAP values and visualization
    """
    # Get SHAP values
    shap_values = explainer.shap_values(instance_df)

    # Create plot
    plt.figure(figsize=(10, 6))

    if plot_type == 'bar':
        shap.summary_plot(shap_values, instance_df, plot_type="bar", show=False)
    elif plot_type == 'beeswarm':
        shap.summary_plot(shap_values, instance_df, show=False)
    else:  # force plot is handled separately
        pass

    plt.tight_layout()

    return shap_values, plt.gcf()

def create_force_plot_html(explainer, shap_values, instance_df):
    """
    Create an interactive HTML force plot
    """
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmpfile:
        if isinstance(shap_values, list):
            # For multi-class, use the prediction class
            predicted_class = 0  # Simplified - would need model prediction
            force_plot = shap.force_plot(
                explainer.expected_value[predicted_class],
                shap_values[predicted_class][0],
                instance_df.iloc[0],
                matplotlib=False,
                show=False
            )
        else:
            # For regression or binary classification
            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                instance_df.iloc[0],
                matplotlib=False,
                show=False
            )

        shap.save_html(tmpfile.name, force_plot)

    with open(tmpfile.name, 'r') as f:
        html_content = f.read()

    os.unlink(tmpfile.name)
    return html_content

def generate_download_link(content, filename):
    """
    Create download link for files
    """
    b64 = base64.b64encode(content).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href
