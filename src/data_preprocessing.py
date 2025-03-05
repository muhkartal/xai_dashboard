"""
app.py

A professional-level Streamlit app that:
1. Loads a trained classification model (e.g., Random Forest).
2. Provides an interactive interface for feature input.
3. Displays predictions and explanation plots (via SHAP).
4. Includes robust logging and caching for performance.

Usage:
------
    streamlit run src/dashboard_app/app.py

Make sure your environment is set up with the required dependencies (see requirements.txt)
and that you have a trained model file (e.g., models/model.pkl).
"""

import os
import sys
import logging
from typing import Dict

import streamlit as st
import pandas as pd
import shap

# If you're using scikit-learn
from sklearn.base import BaseEstimator
import joblib

# If you have a local module for predictions & explanations:
# from src.predict import predict_instance, predict_proba
# from src.explain import init_shap_explainer, explain_prediction

# For demonstration, we'll define minimal placeholders in this file:
def predict_instance(model: BaseEstimator, features: Dict) -> str:
    df = pd.DataFrame([features])
    return str(model.predict(df)[0])

def predict_proba(model: BaseEstimator, features: Dict) -> float:
    df = pd.DataFrame([features])
    return float(model.predict_proba(df).max())

def init_shap_explainer(model: BaseEstimator, background_df: pd.DataFrame = None):
    """
    Initializes and returns a SHAP TreeExplainer for the given model.
    """
    return shap.TreeExplainer(model, data=background_df)

def explain_prediction(explainer, instance_df: pd.DataFrame):
    """
    Returns SHAP values for a single instance.
    """
    return explainer.shap_values(instance_df)

# --------------------------------------------------
# Configure Logging
# --------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --------------------------------------------------
# Configurable Paths (can be read from env or constants)
# --------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
BACKGROUND_DATA_PATH = os.getenv("BACKGROUND_DATA_PATH", "/data/iris.csv")
TARGET_COL = os.getenv("TARGET_COL", "species")  # Example: "species" for Iris dataset


@st.cache_resource
def load_artifacts() -> Dict[str, object]:
    """
    Loads the trained model and sets up the SHAP explainer.
    Leverages Streamlit's caching to avoid re-initializing on every UI interaction.

    :return: A dictionary containing the loaded model and explainer.
    """
    logger.info(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}. Please train and place it there.")
        st.error("Model file not found. Please provide a valid model.pkl.")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Failed to load model. Error: {e}")
        st.stop()

    # SHAP explainer
    explainer = None
    if os.path.exists(BACKGROUND_DATA_PATH):
        try:
            bg_df = pd.read_csv(BACKGROUND_DATA_PATH)
            if TARGET_COL in bg_df.columns:
                bg_df.drop(columns=[TARGET_COL], inplace=True)
            if len(bg_df) > 100:
                # Sample for performance reasons
                bg_df = bg_df.sample(100, random_state=42)
            explainer = init_shap_explainer(model, background_df=bg_df)
            logger.info("SHAP explainer initialized with background data.")
        except Exception as e:
            logger.warning(
                f"Failed to initialize SHAP explainer with background data at {BACKGROUND_DATA_PATH}: {e}. "
                "Will proceed without background data."
            )
            explainer = init_shap_explainer(model)
    else:
        logger.warning(f"Background data not found at {BACKGROUND_DATA_PATH}. Proceeding without it.")
        explainer = init_shap_explainer(model)

    return {"model": model, "explainer": explainer}


def main():
    """
    Main function to render the Streamlit dashboard:
    1) Title & Description
    2) Sidebar inputs (feature sliders)
    3) Prediction & Explanation output
    """
    st.title("Explainable AI (XAI) Dashboard - Classification")

    st.markdown(
        """
        This dashboard demonstrates a classification model with explainable AI features.
        Adjust the feature inputs on the sidebar and click "Predict & Explain" to see
        the model's predictions and corresponding SHAP explanations.
        """
    )

    st.sidebar.header("Model Inputs")

    # Example feature inputs for Iris dataset:
    sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1, 0.1)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5, 0.1)
    petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.sidebar.slider("Petal Width", 0.0, 2.5, 0.2, 0.1)

    # Could add more logic for dynamic or additional features
    input_data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    artifacts = load_artifacts()
    model = artifacts["model"]
    explainer = artifacts["explainer"]

    if st.button("Predict & Explain"):
        # Model prediction
        try:
            pred = predict_instance(model, input_data)
            prob = predict_proba(model, input_data)
            st.write(f"**Prediction:** {pred}")
            st.write(f"**Confidence:** {prob:.2f}")
            logger.info(f"User requested prediction. Got class '{pred}' with confidence {prob:.2f}.")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            st.error(f"Prediction failed. Error: {e}")
            return

        # SHAP Explanation
        if explainer is not None:
            st.subheader("Local Explanation with SHAP")
            instance_df = pd.DataFrame([input_data])
            try:
                shap_values = explain_prediction(explainer, instance_df)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(shap.summary_plot(shap_values, instance_df, plot_type="bar"))
            except Exception as e:
                logger.warning(f"Failed to generate SHAP plot: {e}")
                st.warning("Could not generate SHAP explanation for this instance.")
        else:
            st.warning("No SHAP explainer available. Check logs for errors.")


if __name__ == "__main__":
    main()
