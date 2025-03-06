"""
explain.py

Utilities for model explainability (SHAP).
Includes:
- init_shap_explainer
- explain_prediction
- save_local_explanation_bar
- save_force_plot_html

Author: <Muhkartal / kartal.dev>
"""

import logging
import shap
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def init_shap_explainer(model, background_df: pd.DataFrame = None):
    """
    Initializes a SHAP TreeExplainer for a tree-based model.
    If background_df is large, you might sample it for performance reasons.
    """
    if background_df is not None and len(background_df) > 100:
        background_df = background_df.sample(100, random_state=42)
        logger.info("Sampled background data to 100 rows for performance.")
    explainer = shap.TreeExplainer(model, data=background_df)
    logger.info("SHAP explainer initialized.")
    return explainer

def explain_prediction(explainer, instance_df: pd.DataFrame):
    """
    Generate SHAP values for the given instance(s).
    :return: shap_values (list of arrays if multi-class)
    """
    shap_values = explainer.shap_values(instance_df)
    return shap_values

def save_local_explanation_bar(shap_values, instance_df: pd.DataFrame, filename="shap_local_bar.png"):
    """
    Saves a bar-type summary plot for local explanation as a static PNG.
    """
    shap.summary_plot(shap_values, instance_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"Saved local SHAP bar plot to {filename}")

def save_force_plot_html(explainer, shap_values, instance_df: pd.DataFrame, filename="shap_force.html"):
    """
    Saves an interactive force plot as an HTML file, which you can open in your browser.
    """
    force_html = shap.force_plot(
        explainer.expected_value,
        shap_values,
        instance_df,
        show=False,
        matplotlib=False  # use JavaScript-based visualization
    )
    shap.save_html(filename, force_html)
    logger.info(f"Saved interactive force plot to {filename}. Open it in a browser.")
