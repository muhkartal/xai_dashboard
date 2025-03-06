"""
app.py

Streamlit app allowing users to:
1) Upload a trained classification model file (.pkl or .joblib).
2) Upload a CSV dataset (optional, for global SHAP).
3) Provide a target column name (if needed).
4) Perform local (single-instance) predictions + explanations.
5) Perform global (dataset-level) SHAP explanations.

No local file references or environment variables needed.
All artifacts are user-provided at runtime.

Author: <Your Name>
Date: 2025-03-07
"""

import io
import logging
import sys
from typing import Dict, Optional

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator


# ----------------------------------------------------------------
# Logging Config
# ----------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ----------------------------------------------------------------
# Minimal Predict & SHAP placeholders
# (In real usage, might be in separate 'predict.py' or 'explain.py')
# ----------------------------------------------------------------
def predict_instance(model: BaseEstimator, features: Dict) -> str:
    """
    Predict class for a single feature dictionary. Returns a string label if the model outputs that.
    """
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return str(pred)

def predict_proba(model: BaseEstimator, features: Dict) -> float:
    """
    Returns the max probability for the single instance's predicted class.
    """
    df = pd.DataFrame([features])
    return float(model.predict_proba(df).max())

def init_shap_explainer(model: BaseEstimator, background_df: Optional[pd.DataFrame] = None):
    """
    Creates a SHAP TreeExplainer. If background_df is given, it helps for better global interpretations.
    """
    return shap.TreeExplainer(model, data=background_df)

# ----------------------------------------------------------------
# Helper: Store Model & Data in Session State
# ----------------------------------------------------------------
def load_user_model(uploaded_file) -> Optional[BaseEstimator]:
    """
    Reads a user-uploaded .pkl or .joblib model file and returns the model object.
    """
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            # We can load from a bytes buffer
            model = joblib.load(io.BytesIO(bytes_data))
            logger.info("User model loaded successfully from uploaded file.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from user upload: {e}")
            st.error(f"Failed to load model file: {e}")
    return None

def load_user_csv(uploaded_csv) -> Optional[pd.DataFrame]:
    """
    Reads a user-uploaded CSV file into a pandas DataFrame.
    """
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            logger.info(f"User CSV loaded successfully, shape={df.shape}.")
            return df
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            st.error(f"Failed to read CSV: {e}")
    return None

# ----------------------------------------------------------------
# Pages
# ----------------------------------------------------------------
def page_home():
    """
    Home page: user uploads a model & dataset (optional).
    We store them in st.session_state['model'] and st.session_state['df'].
    """
    st.title("Welcome to the Interactive XAI Dashboard!")
    st.markdown("**Step 1:** Upload your trained model (.pkl or .joblib) and dataset (.csv).")

    # File uploaders
    model_file = st.file_uploader("Upload Model File (.pkl or .joblib)", type=["pkl", "joblib"])
    csv_file = st.file_uploader("Upload Dataset File (.csv)", type=["csv"])

    # If user wants to specify a target column for reference
    st.markdown("**(Optional)** If your dataset has a target column you'd like to drop for background data or reference:")
    target_col_user = st.text_input("Target Column Name (optional)", value="")

    # Load model on user request
    if st.button("Load Artifacts"):
        if model_file:
            st.session_state["model"] = load_user_model(model_file)
        else:
            st.session_state["model"] = None
            st.warning("No model uploaded.")

        if csv_file:
            df = load_user_csv(csv_file)
            st.session_state["df"] = df
            if target_col_user and target_col_user in df.columns:
                st.session_state["target_col"] = target_col_user
                st.success(f"Target column set to '{target_col_user}' for background data usage.")
            else:
                st.session_state["target_col"] = None
        else:
            st.session_state["df"] = None
            st.session_state["target_col"] = None
            st.info("No CSV uploaded, so global SHAP won't be available.")

        if st.session_state.get("model"):
            st.success("Model loaded successfully into session.")
        st.write("**Artifacts loading process complete.**")

def page_local_explanation():
    """
    Allows single-instance predictions and local SHAP explanation if a model is loaded.
    """
    st.title("Local Explanation - Single Instance")
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("No model found in session. Please upload a model on the Home page.")
        return

    model = st.session_state["model"]

    # Feature sliders (example: Iris-like)
    sepal_length = st.slider("Sepal Length", 4.0, 8.5, 5.1, 0.1)
    sepal_width  = st.slider("Sepal Width",  2.0, 4.5,  3.5, 0.1)
    petal_length = st.slider("Petal Length", 1.0, 8.0,  1.4, 0.1)
    petal_width  = st.slider("Petal Width",  0.0, 3.0,  0.2, 0.1)

    input_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    if st.button("Predict & Explain"):
        try:
            pred_class = predict_instance(model, input_data)
            pred_prob = predict_proba(model, input_data)
            st.write(f"**Predicted Class:** {pred_class}")
            st.write(f"**Confidence:** {pred_prob:.2f}")

            # Generate local SHAP
            with st.spinner("Generating SHAP explanation..."):
                # We can init an explainer on the fly with no background data for local only
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(pd.DataFrame([input_data]))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(
                    shap.summary_plot(
                        shap_values,
                        pd.DataFrame([input_data]),
                        plot_type="bar",
                        show=False
                    )
                )
        except Exception as e:
            st.error(f"Prediction/Explanation failed: {e}")

def page_global_explanation():
    """
    Creates a global (dataset-level) SHAP explanation if user has uploaded a CSV and a model.
    """
    st.title("Global Explanation - Dataset Level")

    if "model" not in st.session_state or st.session_state["model"] is None:
        st.warning("No model loaded in session. Please go to Home and upload a model.")
        return
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("No dataset loaded. Please upload a CSV on the Home page.")
        return

    model = st.session_state["model"]
    df_bg = st.session_state["df"].copy()
    target_col = st.session_state["target_col"]  # might be None

    # If user specified a target col, drop it
    if target_col and target_col in df_bg.columns:
        df_bg.drop(columns=[target_col], inplace=True)

    if st.button("Generate Global SHAP Summary"):
        try:
            with st.spinner("Generating global SHAP explanation..."):
                # Possibly sample for performance
                if len(df_bg) > 500:
                    df_bg = df_bg.sample(500, random_state=42)
                    st.info("Sampled dataset to 500 rows for performance.")

                explainer = shap.TreeExplainer(model, data=df_bg)
                shap_values_global = explainer.shap_values(df_bg)

                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(
                    shap.summary_plot(
                        shap_values_global,
                        df_bg,
                        plot_type="bar",
                        show=False
                    )
                )
        except Exception as e:
            st.error(f"Global SHAP generation failed: {e}")

# ----------------------------------------------------------------
# Main Layout
# ----------------------------------------------------------------
def main():
    st.sidebar.title("Navigation")
    pages = {
        "Home": page_home,
        "Local Explanation": page_local_explanation,
        "Global Explanation": page_global_explanation
    }
    choice = st.sidebar.radio("Go to Page:", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
