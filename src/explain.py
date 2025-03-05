import shap
import pandas as pd

def init_shap_explainer(model, background_df: pd.DataFrame = None):
    """
    Initialize a SHAP TreeExplainer for tree-based models.
    """
    # If background_df is large, sample it
    if background_df is not None and len(background_df) > 100:
        background_df = background_df.sample(100, random_state=42)
    explainer = shap.TreeExplainer(model, data=background_df)
    return explainer

def explain_prediction(explainer, instance_df: pd.DataFrame):
    """
    Generate SHAP values for a given instance DataFrame.
    """
    shap_values = explainer.shap_values(instance_df)
    return shap_values
