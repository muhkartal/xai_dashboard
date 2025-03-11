"""
app.py - Upgraded XAI Dashboard

Streamlit app allowing users to:
1) Upload a trained classification model file (.pkl or .joblib).
2) Upload a CSV dataset (optional, for global SHAP).
3) Upload a test instance or create one with UI controls
4) Perform local (single-instance) predictions + explanations.
5) Perform global (dataset-level) SHAP explanations.
6) Compare multiple local predictions side by side
7) Export explanations as PNG or HTML
8) View model metadata and performance metrics

Author: Muhkartal / kartal.dev
Updated: 2025-03-11
"""

import io
import logging
import sys
import datetime
import json
from typing import Dict, Optional, List, Tuple, Union, Any

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import helper functions from utils
from utils.model_utils import load_user_model, detect_model_type, get_model_info
from utils.data_utils import (load_user_csv, get_feature_info, plot_feature_distributions,
                             prepare_sample_instance, get_correlation_matrix)
from utils.shap_utils import (init_shap_explainer, generate_shap_explanation,
                             create_force_plot_html, generate_download_link)
from utils.ui_utils import (apply_custom_styles, display_info_box, create_download_link,
                           display_model_card, render_comparison_view)


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
# Initialization
# ----------------------------------------------------------------
def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "model" not in st.session_state:
        st.session_state["model"] = None
    if "model_info" not in st.session_state:
        st.session_state["model_info"] = None
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "X" not in st.session_state:
        st.session_state["X"] = None
    if "target_col" not in st.session_state:
        st.session_state["target_col"] = None
    if "feature_info" not in st.session_state:
        st.session_state["feature_info"] = {}
    if "explainer" not in st.session_state:
        st.session_state["explainer"] = None
    if "explanations" not in st.session_state:
        st.session_state["explanations"] = []

# ----------------------------------------------------------------
# Pages
# ----------------------------------------------------------------
def page_home():
    """
    Home page: user uploads a model & dataset (optional).
    We store them in st.session_state['model'] and st.session_state['df'].
    """
    st.markdown("<h1 class='main-header'>Interactive XAI Dashboard</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        Welcome to the Interactive Explainable AI Dashboard! This application helps you understand
        how your machine learning models make predictions using SHAP (SHapley Additive exPlanations) values.

        You can:
        * Upload your pre-trained model (.pkl or .joblib) and dataset
        * Visualize global feature importance
        * Make predictions on individual instances
        * Get detailed local explanations
        * Compare multiple explanations side by side
        * Export explanations as images or HTML

        Get started by uploading your model and dataset below.
        """
    )

    # Create two columns for uploaders
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1️⃣ Upload Model")
        model_file = st.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])
        if model_file:
            st.success("✅ Model file uploaded")

    with col2:
        st.markdown("### 2️⃣ Upload Dataset")
        csv_file = st.file_uploader("Upload dataset (.csv)", type=["csv"])
        if csv_file:
            st.success("✅ Dataset uploaded")

        st.markdown("**(Optional)** Specify target column name:")
        target_col_user = st.text_input("Target Column Name", value="")

    # Display load button with a custom style
    if st.button("🚀 Load Artifacts", key="load_btn"):
        with st.spinner("Loading and analyzing artifacts..."):
            if model_file:
                model = load_user_model(model_file)
                if model:
                    st.session_state["model"] = model
                    model_info = get_model_info(model)
                    st.session_state["model_info"] = model_info
                    display_info_box(f"Model loaded: {model_info['model_name']}", "success")
                else:
                    display_info_box("Failed to load model. Please check the file format.", "error")
            else:
                st.session_state["model"] = None
                display_info_box("No model uploaded. Please upload a model file.", "warning")

            if csv_file:
                df = load_user_csv(csv_file)
                if df is not None:
                    st.session_state["df"] = df
                    # Extract feature info
                    feature_info = get_feature_info(df)
                    st.session_state["feature_info"] = feature_info

                    if target_col_user and target_col_user in df.columns:
                        st.session_state["target_col"] = target_col_user
                        # Create X (features) for later SHAP usage
                        X = df.drop(columns=[target_col_user])
                        st.session_state["X"] = X
                        display_info_box(f"Target column set to '{target_col_user}'", "success")
                    else:
                        st.session_state["target_col"] = None
                        st.session_state["X"] = df

                    display_info_box(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns", "success")
                else:
                    display_info_box("Failed to load dataset. Please check the file format.", "error")
            else:
                st.session_state["df"] = None
                st.session_state["X"] = None
                st.session_state["target_col"] = None
                display_info_box("No dataset uploaded. You can still make predictions, but global SHAP and feature distributions won't be available.", "info")

            if st.session_state.get("model") and st.session_state.get("df"):
                # Initialize SHAP explainer if both model and data are available
                with st.spinner("Initializing SHAP explainer..."):
                    background_data = st.session_state.get("X")
                    explainer = init_shap_explainer(st.session_state["model"], background_data)
                    if explainer:
                        st.session_state["explainer"] = explainer
                        display_info_box("SHAP explainer initialized successfully!", "success")
                    else:
                        display_info_box("Failed to initialize SHAP explainer. Some explanation features may not work.", "error")

    # If artifacts have been loaded, show a summary
    if st.session_state.get("model") is not None:
        st.markdown("### 📊 Model Summary")
        model_info = st.session_state.get("model_info", {})

        with st.expander("View Model Details", expanded=True):
            display_model_card(model_info)

        # Show feature importance if available
        if model_info.get('has_feature_importance', False) and st.session_state.get("X") is not None:
            st.markdown("### 📈 Feature Importance")
            with st.expander("View Feature Importance", expanded=True):
                try:
                    feature_names = st.session_state["X"].columns.tolist()
                    model = st.session_state["model"]

                    # Create importance DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    })
                    importance_df = importance_df.sort_values('Importance', ascending=False)

                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax)
                    ax.set_title('Feature Importance (Top 15)')
                    st.pyplot(fig)

                    # Display in table format too
                    st.markdown("#### Feature Importance Table")
                    st.dataframe(importance_df)
                except Exception as e:
                    st.error(f"Error displaying feature importance: {e}")

    # If dataset has been loaded, show a summary
    if st.session_state.get("df") is not None:
        st.markdown("### 📋 Dataset Summary")
        df = st.session_state.get("df")

        with st.expander("View Dataset Statistics", expanded=False):
            st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

            # Display datatypes and basic stats
            st.markdown("#### Data Types and Missing Values")
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing': df.isna().sum(),
                'Missing %': (df.isna().sum() / len(df) * 100).round(2)
            })
            st.dataframe(dtypes_df)

            # Show correlation heatmap if there are numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                st.markdown("#### Correlation Heatmap")
                corr_matrix = get_correlation_matrix(df)
                if corr_matrix is not None:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                    st.pyplot(fig)

def page_local_explanation():
    """
    Allows single-instance predictions and local SHAP explanation if a model is loaded.
    Enhanced with more visualization options and user controls.
    """
    st.markdown("<h1 class='main-header'>Local Explanation - Single Instance</h1>", unsafe_allow_html=True)

    if "model" not in st.session_state or st.session_state["model"] is None:
        display_info_box("No model found in session. Please upload a model on the Home page.", "warning")
        return

    model = st.session_state["model"]
    model_info = st.session_state.get("model_info", {})
    feature_info = st.session_state.get("feature_info", {})

    # Determine available features
    if st.session_state.get("X") is not None:
        available_features = st.session_state["X"].columns.tolist()
    elif feature_info:
        available_features = list(feature_info.keys())
    else:
        # Fallback for custom features (e.g., iris-like)
        available_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Option to upload a test instance
    st.markdown("### 📤 Upload Test Instance or Create Manually")
    test_file = st.file_uploader("Upload a single test instance (CSV with header row)", type=["csv"])

    input_data = {}

    # If user uploads a test file
    if test_file:
        try:
            test_df = pd.read_csv(test_file)
            if len(test_df) > 0:
                st.success(f"Test instance loaded with {test_df.shape[1]} features")
                # Use only the first row if multiple rows
                test_instance = test_df.iloc[0].to_dict()

                # Display the loaded features
                st.markdown("### Loaded Test Instance")
                st.dataframe(test_df.head(1))

                input_data = test_instance
        except Exception as e:
            st.error(f"Error loading test file: {e}")

    # Manual input section
    if not test_file:
        st.markdown("### ✏️ Create Test Instance Manually")

        # Option to load a representative sample if data is available
        if st.session_state.get("df") is not None and st.session_state.get("feature_info"):
            if st.button("Load Representative Sample from Dataset"):
                sample = prepare_sample_instance(st.session_state["df"], st.session_state["feature_info"])
                st.session_state["sample_instance"] = sample
                st.experimental_rerun()

        # If we have a sample instance, use it as default values
        sample_instance = st.session_state.get("sample_instance", {})

        # Create feature inputs based on available features
        feature_values = {}

        # Create columns to display features in a grid
        num_cols = 2  # Number of columns in the grid
        features_per_col = len(available_features) // num_cols + (1 if len(available_features) % num_cols > 0 else 0)

        cols = st.columns(num_cols)

        for i, feature in enumerate(available_features):
            col_idx = i // features_per_col

            with cols[col_idx]:
                # Check if we have info about this feature
                if feature in feature_info:
                    info = feature_info[feature]
                    # For numeric features
                    if 'min' in info and 'max' in info and info['min'] is not None and info['max'] is not None:
                        min_val = float(info['min'])
                        max_val = float(info['max'])
                        step = (max_val - min_val) / 100 if max_val > min_val else 0.1
                        default_val = float(sample_instance.get(feature, info.get('median', (min_val + max_val) / 2)))
                        feature_values[feature] = st.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            step=step
                        )
                    # For categorical features
                    elif 'unique_values' in info and isinstance(info['unique_values'], list):
                        options = info['unique_values']
                        default_idx = 0
                        if feature in sample_instance and sample_instance[feature] in options:
                            default_idx = options.index(sample_instance[feature])
                        feature_values[feature] = st.selectbox(
                            f"{feature}",
                            options=options,
                            index=default_idx
                        )
                    else:
                        # Generic text input fallback
                        default_value = sample_instance.get(feature, "")
                        feature_values[feature] = st.text_input(f"{feature}", value=default_value)
                else:
                    # Generic numerical slider fallback
                    default_val = float(sample_instance.get(feature, 5.0))
                    feature_values[feature] = st.slider(f"{feature}", 0.0, 10.0, default_val, 0.1)

        input_data = feature_values

    # Visualization options
    st.markdown("### 🎨 Visualization Options")
    vis_cols = st.columns(2)

    with vis_cols[0]:
        plot_type = st.radio(
            "SHAP Plot Type:",
            options=["Bar", "Beeswarm", "Force"],
            horizontal=True
        )

    with vis_cols[1]:
        export_format = st.radio(
            "Export Format:",
            options=["PNG", "HTML", "CSV"],
            horizontal=True
        )

    # Button to trigger prediction and explanation
    if st.button("🔮 Predict & Explain", key="predict_btn"):
        if not input_data:
            display_info_box("No input data available. Please create or upload a test instance.", "error")
            return

        try:
            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_data])

            # Make prediction
            model_type, _ = detect_model_type(model)

            if model_type == 'classification':
                pred_result = model.predict(input_df)[0]
                # Try to get probability if available
                pred_prob = None
                if hasattr(model, 'predict_proba'):
                    pred_prob = float(model.predict_proba(input_df).max())
            else:  # regression
                pred_result = float(model.predict(input_df)[0])
                pred_prob = None

            # Display prediction result
            result_cols = st.columns(2)
            with result_cols[0]:
                st.markdown("### 🎯 Prediction Result")
                if model_type == 'classification':
                    st.markdown(f"**Predicted Class:** {pred_result}")
                    if pred_prob is not None:
                        st.markdown(f"**Confidence:** {pred_prob:.2f}")
                else:  # regression
                    st.markdown(f"**Predicted Value:** {pred_result:.4f}")

            # SHAP explanation
            with st.spinner("Generating SHAP explanation..."):
                # Initialize explainer if needed
                if "explainer" not in st.session_state or st.session_state["explainer"] is None:
                    # Create an explainer with no background data
                    explainer = init_shap_explainer(model)
                    st.session_state["explainer"] = explainer
                    if explainer is None:
                        display_info_box("Could not initialize SHAP explainer. Please check your model.", "error")
                        return

                explainer = st.session_state["explainer"]

                # Get SHAP values
                shap_values = explainer.shap_values(input_df)

                # Store explanation in history for comparison page
                if "explanations" not in st.session_state:
                    st.session_state["explanations"] = []

                # Add to explanations with timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                explanation_data = {
                    "timestamp": timestamp,
                    "input_data": input_data,
                    "prediction": pred_result,
                    "probability": pred_prob,
                    "shap_values": shap_values,
                }

                # Keep only the most recent explanations
                st.session_state["explanations"].append(explanation_data)
                if len(st.session_state["explanations"]) > 10:
                    st.session_state["explanations"] = st.session_state["explanations"][-10:]

                # Display SHAP plots based on user selection
                st.markdown("### 📊 SHAP Explanation")

                if plot_type == "Bar" or plot_type == "Beeswarm":
                    shap_plot_type = 'bar' if plot_type == "Bar" else 'beeswarm'
                    # Generate explanation plot
                    shap_values, fig = generate_shap_explanation(explainer, input_df, plot_type=shap_plot_type)
                    st.pyplot(fig)

                    # Save figure to bytes for download
                    img_data = io.BytesIO()
                    fig.savefig(img_data, format='png', bbox_inches='tight')
                    img_data.seek(0)

                elif plot_type == "Force":
                    # Generate force plot HTML
                    html_content = create_force_plot_html(explainer, shap_values, input_df)
                    st.components.v1.html(html_content, height=200)

                    # Store HTML for download
                    html_bytes = html_content.encode()

                # Export options
                st.markdown("### 💾 Export Options")

                if export_format == "PNG" and plot_type in ["Bar", "Beeswarm"]:
                    st.markdown(create_download_link(img_data.getvalue(), "shap_explanation.png"), unsafe_allow_html=True)

                elif export_format == "HTML" and plot_type == "Force":
                    st.markdown(create_download_link(html_bytes, "shap_force_plot.html"), unsafe_allow_html=True)

                elif export_format == "CSV":
                    # Create a CSV with features and their SHAP values
                    shap_df = pd.DataFrame()
                    shap_df['Feature'] = input_df.columns

                    if isinstance(shap_values, list):  # Multi-class
                        # Get the predicted class
                        if model_type == 'classification' and hasattr(model, 'classes_'):
                            try:
                                class_idx = list(model.classes_).index(pred_result)
                                class_shap = shap_values[class_idx][0]
                                shap_df['SHAP Value'] = class_shap
                            except (ValueError, IndexError):
                                # Fallback if class not found
                                shap_df['SHAP Value'] = shap_values[0][0]
                        else:
                            # For regression or if class not found
                            shap_df['SHAP Value'] = shap_values[0]
                    else:
                        shap_df['SHAP Value'] = shap_values[0]

                    csv_bytes = shap_df.to_csv(index=False).encode()
                    st.markdown(create_download_link(csv_bytes, "shap_values.csv"), unsafe_allow_html=True)

                # Success message
                display_info_box("Explanation generated and saved to history for comparison!", "success")

        except Exception as e:
            display_info_box(f"Error generating explanation: {e}", "error")
            logger.error(f"Error in local explanation: {e}", exc_info=True)



def page_train_model():
    """
    New page that allows users to train machine learning models directly in the app.
    - Upload a dataset
    - Select features and target
    - Choose model type and hyperparameters
    - Train model with cross-validation
    - Save the model
    """
    st.markdown("<h1 class='main-header'>Train Your Own Model</h1>", unsafe_allow_html=True)

    # Step 1: Data Upload and Preparation
    st.markdown("### 1️⃣ Upload Training Data")

    train_file = st.file_uploader("Upload CSV dataset for training", type=["csv"])

    if train_file is None:
        display_info_box("Please upload a CSV file to start the training process.", "info")
        return

    # Load the dataset
    try:
        with st.spinner("Loading and analyzing dataset..."):
            df = pd.read_csv(train_file)
            st.success(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

            # Show a preview of the data
            st.markdown("### Data Preview")
            st.dataframe(df.head())

            # Data information
            st.markdown("### Data Information")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

            # Check for missing values
            missing_vals = df.isnull().sum()
            if missing_vals.sum() > 0:
                st.warning("Dataset contains missing values. Consider preprocessing.")
                st.write(missing_vals[missing_vals > 0])
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Step 2: Feature and Target Selection
    st.markdown("### 2️⃣ Select Features and Target")

    # Target column selection
    target_col = st.selectbox("Select Target Column", df.columns.tolist())

    # Feature selection
    available_features = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect(
        "Select Features to Use",
        available_features,
        default=available_features
    )

    if len(selected_features) == 0:
        display_info_box("Please select at least one feature column.", "warning")
        return

    # Create X and y
    X = df[selected_features]
    y = df[target_col]

    # Determine problem type
    unique_target_values = y.nunique()

    if pd.api.types.is_numeric_dtype(y):
        if unique_target_values <= 10:
            problem_type = st.radio(
                "Problem Type (Detected)",
                ["Classification", "Regression"],
                index=0
            )
        else:
            problem_type = st.radio(
                "Problem Type (Detected)",
                ["Classification", "Regression"],
                index=1
            )
    else:
        problem_type = "Classification"
        st.info(f"Classification problem detected: {unique_target_values} unique classes")

    # Step 3: Select Model and Hyperparameters
    st.markdown("### 3️⃣ Select Model Type")

    if problem_type == "Classification":
        model_options = [
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression",
            "Support Vector Machine",
            "XGBoost"
        ]
    else:  # Regression
        model_options = [
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression",
            "Support Vector Machine",
            "XGBoost"
        ]

    selected_model = st.selectbox("Select Model Type", model_options)

    # Show different hyperparameter options based on the selected model
    with st.expander("Model Hyperparameters", expanded=True):
        if selected_model in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            n_estimators = st.slider("Number of Estimators", min_value=10, max_value=500, value=100, step=10)
            max_depth = st.slider("Max Depth", min_value=1, max_value=30, value=10)

            if selected_model == "XGBoost":
                learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                subsample = st.slider("Subsample Ratio", min_value=0.5, max_value=1.0, value=0.8, step=0.1)

        elif selected_model in ["Logistic Regression", "Linear Regression"]:
            C = st.slider("Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            max_iter = st.slider("Maximum Iterations", min_value=100, max_value=2000, value=1000, step=100)

        elif selected_model == "Support Vector Machine":
            C = st.slider("Regularization Parameter (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])

    # Step 4: Training Options
    st.markdown("### 4️⃣ Training Settings")

    test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20) / 100
    cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5)
    random_state = st.number_input("Random State", value=42)

    # Option for preprocessing
    do_preprocessing = st.checkbox("Apply Automatic Preprocessing", value=True)

    # Step 5: Train Model Button
    if st.button("🚀 Train Model", key="train_model_btn"):
        try:
            with st.spinner("Training model, please wait..."):
                # Import necessary libraries
                from sklearn.model_selection import train_test_split, cross_val_score
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import Pipeline
                from sklearn.impute import SimpleImputer
                from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix
                import numpy as np
                import time

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

                # Determine categorical and numerical columns
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

                # Create preprocessing pipeline
                preprocessor = None
                if do_preprocessing:
                    numerical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])

                    if categorical_cols:
                        categorical_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))
                        ])

                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', numerical_transformer, numerical_cols),
                                ('cat', categorical_transformer, categorical_cols)
                            ]
                        )
                    else:
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', numerical_transformer, numerical_cols)
                            ]
                        )

                # Initialize model based on selection
                if selected_model == "Random Forest":
                    if problem_type == "Classification":
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth if max_depth > 0 else None,
                            random_state=random_state
                        )
                    else:
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth if max_depth > 0 else None,
                            random_state=random_state
                        )

                elif selected_model == "Gradient Boosting":
                    if problem_type == "Classification":
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )
                    else:
                        from sklearn.ensemble import GradientBoostingRegressor
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=random_state
                        )

                elif selected_model == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        random_state=random_state
                    )

                elif selected_model == "Linear Regression":
                    from sklearn.linear_model import Ridge
                    model = Ridge(
                        alpha=1/C,
                        max_iter=max_iter,
                        random_state=random_state
                    )

                elif selected_model == "Support Vector Machine":
                    if problem_type == "Classification":
                        from sklearn.svm import SVC
                        model = SVC(
                            C=C,
                            kernel=kernel,
                            probability=True,
                            random_state=random_state
                        )
                    else:
                        from sklearn.svm import SVR
                        model = SVR(
                            C=C,
                            kernel=kernel
                        )

                elif selected_model == "XGBoost":
                    try:
                        import xgboost as xgb
                        if problem_type == "Classification":
                            model = xgb.XGBClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                random_state=random_state,
                                use_label_encoder=False,
                                eval_metric='logloss'
                            )
                        else:
                            model = xgb.XGBRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                subsample=subsample,
                                random_state=random_state
                            )
                    except ImportError:
                        st.error("XGBoost is not installed. Please install it or choose another model.")
                        return

                # Create pipeline with preprocessing if enabled
                if preprocessor:
                    pipeline = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('model', model)
                    ])
                else:
                    pipeline = model

                # Calculate training start time
                start_time = time.time()

                # Cross-validation
                if problem_type == "Classification":
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='accuracy')
                    score_name = "Accuracy"
                else:
                    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
                    score_name = "Neg MSE"

                # Train the final model on the full training set
                pipeline.fit(X_train, y_train)

                # Make predictions on test set
                y_pred = pipeline.predict(X_test)

                # Calculate training time
                training_time = time.time() - start_time

                # Display training results
                st.markdown("### 🎯 Training Results")

                # Cross-validation results
                st.markdown(f"**{cv_folds}-Fold Cross-Validation Results:**")
                cv_df = pd.DataFrame({
                    'Fold': range(1, len(cv_scores) + 1),
                    f'{score_name}': cv_scores
                })

                # Display cross-validation results
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(cv_df)

                with col2:
                    fig, ax = plt.subplots()
                    ax.bar(cv_df['Fold'], cv_df[score_name])
                    ax.set_xlabel('Fold')
                    ax.set_ylabel(score_name)
                    ax.set_title(f'Cross-Validation {score_name}')
                    st.pyplot(fig)

                st.markdown(f"**Mean CV {score_name}:** {cv_scores.mean():.4f}")
                st.markdown(f"**Training Time:** {training_time:.2f} seconds")

                # Test set performance
                st.markdown("### 📊 Test Set Performance")

                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    # For multiclass classification
                    if len(np.unique(y)) > 2:
                        st.markdown(f"**Accuracy:** {accuracy:.4f}")
                        st.markdown(f"**F1 Score (weighted):** {f1:.4f}")
                    else:  # Binary classification
                        from sklearn.metrics import precision_score, recall_score, roc_auc_score
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)

                        # Get probability predictions for ROC AUC
                        try:
                            y_prob = pipeline.predict_proba(X_test)[:, 1]
                            auc = roc_auc_score(y_test, y_prob)
                            st.markdown(f"**AUC:** {auc:.4f}")
                        except:
                            pass

                        st.markdown(f"**Accuracy:** {accuracy:.4f}")
                        st.markdown(f"**Precision:** {precision:.4f}")
                        st.markdown(f"**Recall:** {recall:.4f}")
                        st.markdown(f"**F1 Score:** {f1:.4f}")

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)

                else:  # Regression
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)

                    st.markdown(f"**Mean Squared Error:** {mse:.4f}")
                    st.markdown(f"**Root Mean Squared Error:** {rmse:.4f}")
                    st.markdown(f"**R² Score:** {r2:.4f}")

                    # Actual vs Predicted Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax.set_xlabel('Actual')
                    ax.set_ylabel('Predicted')
                    ax.set_title('Actual vs Predicted')
                    st.pyplot(fig)

                # Feature Importance (if applicable)
                if hasattr(model, 'feature_importances_') or (hasattr(pipeline, 'named_steps') and hasattr(pipeline.named_steps.get('model', None), 'feature_importances_')):
                    st.markdown("### 🔍 Feature Importance")

                    # Get feature importances
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = selected_features
                    else:
                        importances = pipeline.named_steps['model'].feature_importances_

                        # Handle case where preprocessing changes feature names (e.g., one-hot encoding)
                        if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                            feature_names = preprocessor.get_feature_names_out()
                        else:
                            feature_names = selected_features

                    # Create feature importance DataFrame
                    if len(importances) == len(feature_names):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)

                        # Plot feature importances
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                        ax.set_title('Feature Importance')
                        st.pyplot(fig)

                        # Display as table
                        st.dataframe(importance_df)

                # Save model options
                st.markdown("### 💾 Save Trained Model")

                model_filename = st.text_input("Model Filename", value=f"{selected_model.lower().replace(' ', '_')}_{problem_type.lower()}.joblib")

                if st.button("Save Model"):
                    try:
                        # Save model to BytesIO
                        import joblib

                        model_bytes = io.BytesIO()
                        joblib.dump(pipeline, model_bytes)
                        model_bytes.seek(0)

                        # Create download link
                        st.download_button(
                            label="Download Model File",
                            data=model_bytes,
                            file_name=model_filename,
                            mime="application/octet-stream"
                        )

                        # Save to session state for immediate use
                        st.session_state["model"] = pipeline

                        # Extract model info
                        model_info = {
                            'model_name': selected_model,
                            'model_type': problem_type,
                            'num_classes': len(np.unique(y)) if problem_type == "Classification" else None,
                            'params': pipeline.get_params() if hasattr(pipeline, 'get_params') else {},
                            'has_feature_importance': hasattr(model, 'feature_importances_') or (hasattr(pipeline, 'named_steps') and hasattr(pipeline.named_steps.get('model', None), 'feature_importances_'))
                        }
                        st.session_state["model_info"] = model_info

                        # Save feature info for immediate use
                        if st.session_state.get("X") is None:
                            st.session_state["X"] = X
                            st.session_state["feature_info"] = get_feature_info(X)

                        # Initialize SHAP explainer for immediate use
                        with st.spinner("Initializing SHAP explainer..."):
                            explainer = init_shap_explainer(pipeline, X.sample(min(100, len(X))))
                            if explainer:
                                st.session_state["explainer"] = explainer

                        st.success(f"Model saved and loaded into the app! You can now use it for explanations.")
                    except Exception as e:
                        st.error(f"Error saving model: {e}")

        except Exception as e:
            st.error(f"Error during model training: {e}")
            logger.error(f"Training error: {e}", exc_info=True)




def page_global_explanation():
    """
    Creates a global (dataset-level) SHAP explanation if user has uploaded a CSV and a model.
    Enhanced with more visualization options and sampling controls.
    """
    st.markdown("<h1 class='main-header'>Global Explanation - Dataset Level</h1>", unsafe_allow_html=True)

    if "model" not in st.session_state or st.session_state["model"] is None:
        display_info_box("No model loaded in session. Please go to Home and upload a model.", "warning")
        return
    if "X" not in st.session_state or st.session_state["X"] is None:
        display_info_box("No dataset loaded. Please upload a CSV on the Home page.", "warning")
        return

    model = st.session_state["model"]
    X = st.session_state["X"]

    # Settings for global SHAP
    st.markdown("### ⚙️ Global SHAP Settings")

    settings_cols = st.columns(2)

    with settings_cols[0]:
        sample_size = st.slider(
            "Number of samples to use (more = slower but more accurate):",
            min_value=50,
            max_value=min(1000, len(X)),
            value=min(200, len(X)),
            step=50
        )

        plot_type = st.radio(
            "Plot Type:",
            options=["Summary Bar", "Summary Dot", "Dependence"],
            horizontal=True
        )

    with settings_cols[1]:
        if plot_type == "Dependence":
            available_features = X.columns.tolist()
            feature_for_dependence = st.selectbox(
                "Feature for Dependence Plot:",
                options=available_features
            )

        # Option to use custom dataset
        use_custom_data = st.checkbox("Use specific dataset subset")
        if use_custom_data:
            st.info("You can select specific conditions to filter the data for analysis.")
            # This could be expanded with more complex filtering options

    # Generate global SHAP button
    if st.button("📊 Generate Global SHAP Explanation", key="global_shap_btn"):
        try:
            with st.spinner("Generating global SHAP explanation..."):
                # Possibly sample for performance
                if len(X) > sample_size:
                    sampled_X = X.sample(sample_size, random_state=42)
                    st.info(f"Sampled dataset to {sample_size} rows for performance.")
                else:
                    sampled_X = X

                # Initialize explainer if needed
                if "explainer" not in st.session_state or st.session_state["explainer"] is None:
                    explainer = init_shap_explainer(model, sampled_X)
                    st.session_state["explainer"] = explainer
                    if explainer is None:
                        display_info_box("Could not initialize SHAP explainer. Please check your model.", "error")
                        return

                explainer = st.session_state["explainer"]

                # Get SHAP values
                shap_values = explainer.shap_values(sampled_X)

                # Create visualization based on selected plot type
                st.markdown("### 📈 Global SHAP Explanation")

                if plot_type == "Summary Bar":
                    fig = plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, sampled_X, plot_type="bar", show=False)
                    st.pyplot(fig)

                elif plot_type == "Summary Dot":
                    fig = plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, sampled_X, show=False)
                    st.pyplot(fig)

                elif plot_type == "Dependence":
                    fig = plt.figure(figsize=(10, 6))
                    # For multi-class, use first class for simplicity
                    if isinstance(shap_values, list):
                        # Use first class for simplicity in this example
                        shap.dependence_plot(
                            feature_for_dependence,
                            shap_values[0],
                            sampled_X,
                            show=False
                        )
                    else:
                        shap.dependence_plot(
                            feature_for_dependence,
                            shap_values,
                            sampled_X,
                            show=False
                        )
                    st.pyplot(fig)

                # Export options
                st.markdown("### 💾 Export Options")

                # Save figure to bytes for download
                img_data = io.BytesIO()
                fig.savefig(img_data, format='png', bbox_inches='tight', dpi=300)
                img_data.seek(0)
                st.markdown(create_download_link(img_data.getvalue(), "global_shap.png"), unsafe_allow_html=True)

                # Also provide detailed SHAP data as CSV
                if isinstance(shap_values, list):
                    # For multi-class, use mean absolute SHAP value across classes
                    shap_mean = np.abs(np.mean([shap_values[i] for i in range(len(shap_values))], axis=0))
                    mean_vals = np.mean(np.abs(shap_mean), axis=0)
                else:
                    mean_vals = np.mean(np.abs(shap_values), axis=0)

                shap_summary = pd.DataFrame({
                    'Feature': X.columns,
                    'Mean |SHAP|': mean_vals
                }).sort_values('Mean |SHAP|', ascending=False)

                st.markdown("### Feature Importance based on SHAP values")
                st.dataframe(shap_summary)

                # Provide CSV download
                csv_bytes = shap_summary.to_csv(index=False).encode()
                st.markdown(create_download_link(csv_bytes, "global_shap_values.csv"), unsafe_allow_html=True)

        except Exception as e:
            display_info_box(f"Global SHAP generation failed: {e}", "error")
            logger.error(f"Error in global explanation: {e}", exc_info=True)

def page_comparison_view():
    """
    New page for comparing multiple explanations side by side.
    """
    st.markdown("<h1 class='main-header'>Explanation Comparison</h1>", unsafe_allow_html=True)

    if "explanations" not in st.session_state or not st.session_state["explanations"]:
        display_info_box("No explanations available for comparison. Generate some explanations in the Local Explanation page first.", "warning")
        return

    explanations = st.session_state["explanations"]

    # Get feature names from the most recent explanation
    if st.session_state.get("X") is not None:
        feature_names = st.session_state["X"].columns.tolist()
    else:
        # Try to extract feature names from the most recent explanation
        latest_exp = explanations[-1]
        feature_names = list(latest_exp.get("input_data", {}).keys())

    # Render the comparison view
    render_comparison_view(explanations, feature_names)

    # Option to clear history
    if st.button("🗑️ Clear Explanation History"):
        st.session_state["explanations"] = []
        st.success("Explanation history cleared.")
        st.experimental_rerun()

def page_model_summary():
    """
    New page showing detailed model information and metadata.
    """
    st.markdown("<h1 class='main-header'>Model Analysis & Summary</h1>", unsafe_allow_html=True)

    if "model" not in st.session_state or st.session_state["model"] is None:
        display_info_box("No model loaded in session. Please go to Home and upload a model.", "warning")
        return

    model = st.session_state["model"]
    model_info = st.session_state.get("model_info", {})

    # Display comprehensive model information
    st.markdown("### 📋 Model Metadata")

    # Basic model info
    display_model_card(model_info)

    # Technical details
    st.markdown("### 🔍 Technical Details")

    # Try to get more technical details depending on model type
    model_type = type(model).__name__

    if "sklearn" in str(type(model)):
        with st.expander("View scikit-learn Model Details", expanded=True):
            # Display sklearn-specific attributes
            st.markdown("#### Model Parameters")
            params = model.get_params()

            # Format parameters as a dataframe for better viewing
            param_df = pd.DataFrame([
                {"Parameter": param, "Value": str(value)}
                for param, value in params.items()
            ])
            st.dataframe(param_df)

            # Try to show feature importances
            if hasattr(model, 'feature_importances_') and st.session_state.get("X") is not None:
                st.markdown("#### Feature Importance")

                feature_names = st.session_state["X"].columns.tolist()
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)

                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)

                # Show as table too
                st.dataframe(importance_df)

            # For tree-based models, offer tree visualization
            if hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'tree_'):
                st.markdown("#### Sample Tree Visualization")
                st.info("Showing the first tree in the ensemble (limited view for complex models)")

                # Only import these if needed
                from sklearn.tree import plot_tree

                # Plot the first tree with limited depth for visualization
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_tree(
                    model.estimators_[0],
                    max_depth=3,
                    feature_names=st.session_state["X"].columns if st.session_state.get("X") is not None else None,
                    filled=True,
                    ax=ax
                )
                st.pyplot(fig)

    elif "xgboost" in str(type(model)):
        with st.expander("View XGBoost Model Details", expanded=True):
            # Display XGBoost-specific attributes
            st.markdown("#### Model Parameters")

            # Get params
            params = model.get_params()
            param_df = pd.DataFrame([
                {"Parameter": param, "Value": str(value)}
                for param, value in params.items()
            ])
            st.dataframe(param_df)

            # Feature importance
            if hasattr(model, 'feature_importances_') and st.session_state.get("X") is not None:
                st.markdown("#### Feature Importance")

                feature_names = st.session_state["X"].columns.tolist()

                # Get both weight (F score) and gain importance types
                importance_types = ['weight', 'gain', 'cover']

                for imp_type in importance_types:
                    try:
                        importance = model.get_booster().get_score(importance_type=imp_type)
                        # Convert to dataframe for visualization
                        imp_df = pd.DataFrame({
                            'Feature': list(importance.keys()),
                            f'Importance ({imp_type})': list(importance.values())
                        })
                        imp_df = imp_df.sort_values(f'Importance ({imp_type})', ascending=False)

                        st.markdown(f"**Importance Type: {imp_type}**")

                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x=f'Importance ({imp_type})', y='Feature', data=imp_df.head(15), ax=ax)
                        ax.set_title(f'Feature Importance ({imp_type})')
                        st.pyplot(fig)
                    except:
                        pass

    # If we have a dataset loaded, show cross-validated performance metrics
    if st.session_state.get("X") is not None and st.session_state.get("target_col") is not None:
        st.markdown("### 📏 Model Performance Evaluation")

        with st.expander("Run Cross-Validation Performance Analysis", expanded=False):
            if st.button("Run Cross-Validation"):
                try:
                    from sklearn.model_selection import cross_val_score, KFold
                    from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, r2_score

                    X = st.session_state["X"]
                    df = st.session_state["df"]
                    target_col = st.session_state["target_col"]
                    y = df[target_col]

                    with st.spinner("Running cross-validation..."):
                        # Setup cross-validation
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)

                        # Determine metrics based on model type
                        model_type, _ = detect_model_type(model)

                        if model_type == 'classification':
                            metrics = {
                                'Accuracy': make_scorer(accuracy_score),
                                'F1': make_scorer(f1_score, average='weighted')
                            }
                        else:  # regression
                            metrics = {
                                'MSE': make_scorer(mean_squared_error, greater_is_better=False),
                                'R²': make_scorer(r2_score)
                            }

                        # Run cross-validation for each metric
                        results = {}
                        for metric_name, scorer in metrics.items():
                            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
                            results[metric_name] = scores

                        # Display results
                        st.markdown("#### Cross-Validation Results (5-fold)")

                        # Plot results
                        fig, ax = plt.subplots(figsize=(10, 6))

                        data = []
                        for metric, scores in results.items():
                            for fold, score in enumerate(scores):
                                data.append({
                                    'Metric': metric,
                                    'Fold': f'Fold {fold+1}',
                                    'Score': score
                                })

                        results_df = pd.DataFrame(data)
                        sns.barplot(x='Fold', y='Score', hue='Metric', data=results_df, ax=ax)
                        ax.set_title('Cross-Validation Performance')
                        st.pyplot(fig)

                        # Show summary statistics
                        summary = pd.DataFrame({
                            'Metric': list(results.keys()),
                            'Mean': [scores.mean() for scores in results.values()],
                            'Std Dev': [scores.std() for scores in results.values()],
                            'Min': [scores.min() for scores in results.values()],
                            'Max': [scores.max() for scores in results.values()],
                        })
                        st.dataframe(summary)

                except Exception as e:
                    st.error(f"Error running cross-validation: {e}")

# ----------------------------------------------------------------
# Main App
# ----------------------------------------------------------------
def main():
    # Configure page
    st.set_page_config(
        page_title="XAI Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styles
    apply_custom_styles()

    # Initialize session state
    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("XAI Dashboard")
    st.sidebar.markdown("### Navigation")

    # Navigation options
    page_options = [
        "Home",
        "Local Explanation",
        "Train Model",
        "Global Explanation",
        "Comparison View",
        "Model Summary"
    ]

    page = st.sidebar.selectbox("Go to Page:", page_options)

    # Additional sidebar info
    with st.sidebar.expander("About this App", expanded=False):
        st.markdown("""
        ### XAI Dashboard

        This dashboard helps you understand how your machine learning models make predictions using SHAP (SHapley Additive exPlanations) values.

        **Features:**
        - Upload your own model (.pkl/.joblib)
        - Upload your dataset (.csv)
        - Generate local explanations for individual predictions
        - Create global explanations to understand overall model behavior
        - Compare multiple explanations side by side
        - Export visualizations for your reports

        **Author:** Muhkartal / kartal.dev
        """)

    # Display session info in sidebar
    with st.sidebar.expander("Session Status", expanded=False):
        if st.session_state.get("model") is not None:
            model_info = st.session_state.get("model_info", {})
            st.success(f"✅ Model loaded: {model_info.get('model_name', 'Unknown')}")
        else:
            st.warning("❌ No model loaded")

        if st.session_state.get("df") is not None:
            df = st.session_state.get("df")
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.warning("❌ No dataset loaded")

        if st.session_state.get("explainer") is not None:
            st.success("✅ SHAP explainer initialized")
        else:
            st.warning("❌ SHAP explainer not initialized")

    # Display appropriate page
    if page == "Home":
        page_home()
    elif page == "Local Explanation":
        page_local_explanation()
    elif page == "Train Model":
        page_train_model()
    elif page == "Global Explanation":
        page_global_explanation()
    elif page == "Comparison View":
        page_comparison_view()
    elif page == "Model Summary":
        page_model_summary()


if __name__ == "__main__":
    main()
