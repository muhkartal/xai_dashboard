import streamlit as st
import base64
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def apply_custom_styles():
    """Apply custom CSS styling to the app."""
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        color: #3366FF;
        margin-bottom: 1rem;
    }

    .section-header {
        font-size: 1.8rem;
        color: #343A40;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Info boxes */
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #2196F3;
    }

    .warning-box {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FFC107;
    }

    .success-box {
        background-color: #D4EDDA;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #28A745;
    }

    .error-box {
        background-color: #F8D7DA;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #DC3545;
    }

    /* Feature input styling */
    .feature-input-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Button styling */
    .primary-button {
        background-color: #3366FF;
        color: white;
        font-weight: bold;
    }

    .secondary-button {
        background-color: #6C757D;
        color: white;
    }

    /* Table styling */
    .styled-table th {
        background-color: #F8F9FA;
    }

    .styled-table tr:nth-child(even) {
        background-color: #F8F9FA;
    }

    /* Card styling */
    .card {
        border: 1px solid #DEE2E6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def display_info_box(message: str, box_type: str = "info"):
    """
    Display a styled info box with custom message.

    Args:
        message: The text to display
        box_type: One of "info", "warning", "success", "error"
    """
    box_classes = {
        "info": "info-box",
        "warning": "warning-box",
        "success": "success-box",
        "error": "error-box"
    }

    box_class = box_classes.get(box_type, "info-box")
    st.markdown(f'<div class="{box_class}">{message}</div>', unsafe_allow_html=True)

def create_download_link(content, filename: str, text: str = None):
    """
    Create a download link for any content.

    Args:
        content: The binary content to download
        filename: Name for the downloaded file
        text: Optional link text (defaults to "Download {filename}")
    """
    b64 = base64.b64encode(content).decode()
    if text is None:
        text = f"Download {filename}"
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_model_card(model_info: Dict[str, Any]):
    """
    Display model information in a card format.
    """
    model_type = model_info.get('model_type', 'Unknown')
    model_name = model_info.get('model_name', 'Unknown')

    # Header and main info
    st.markdown(f"<h3>{model_name}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Task Type:</strong> {model_type.capitalize()}</p>", unsafe_allow_html=True)

    # Show class info for classification
    if model_type == 'classification' and 'num_classes' in model_info:
        num_classes = model_info['num_classes']
        st.markdown(f"<p><strong>Number of Classes:</strong> {num_classes}</p>", unsafe_allow_html=True)

    # Hyperparameters
    if 'params' in model_info and model_info['params']:
        params = model_info['params']
        st.markdown("<p><strong>Key Hyperparameters:</strong></p>", unsafe_allow_html=True)

        # Filter to show most important params
        important_params = ["n_estimators", "max_depth", "learning_rate", "C", "kernel",
                           "gamma", "solver", "alpha", "subsample", "colsample_bytree"]

        # Display params in a compact format
        param_strs = []
        for param in important_params:
            if param in params:
                param_strs.append(f"<strong>{param}:</strong> {params[param]}")

        # Show params in columns if there are many
        if len(param_strs) > 3:
            cols = st.columns(2)
            half = len(param_strs) // 2 + len(param_strs) % 2

            with cols[0]:
                for param_str in param_strs[:half]:
                    st.markdown(f"<p>{param_str}</p>", unsafe_allow_html=True)

            with cols[1]:
                for param_str in param_strs[half:]:
                    st.markdown(f"<p>{param_str}</p>", unsafe_allow_html=True)
        else:
            for param_str in param_strs:
                st.markdown(f"<p>{param_str}</p>", unsafe_allow_html=True)

def render_comparison_view(explanations: List[Dict], feature_names: List[str]):
    """
    Create a side-by-side comparison of multiple explanations.

    Args:
        explanations: List of explanation dictionaries
        feature_names: List of feature names
    """
    if not explanations:
        display_info_box("No explanations available for comparison. Generate some explanations first.", "warning")
        return

    # Select explanations to compare
    st.markdown("### Select Explanations to Compare")

    # Create a selection interface
    options = [f"{i+1}. {exp.get('timestamp', 'Unknown')} - Prediction: {exp.get('prediction', 'Unknown')}"
              for i, exp in enumerate(explanations)]

    selected_indices = st.multiselect(
        "Choose explanations to compare (max 3):",
        options=range(len(options)),
        format_func=lambda i: options[i],
        default=list(range(min(3, len(options))))
    )

    if not selected_indices:
        display_info_box("Please select at least one explanation to view.", "info")
        return

    if len(selected_indices) > 3:
        display_info_box("More than 3 explanations selected. Only the first 3 will be displayed.", "warning")
        selected_indices = selected_indices[:3]

    # Get the selected explanations
    selected_exps = [explanations[i] for i in selected_indices]

    # Display comparison
    st.markdown("### Explanation Comparison")

    # Side-by-side comparison
    cols = st.columns(len(selected_exps))

    for i, (col, exp) in enumerate(zip(cols, selected_exps)):
        with col:
            st.markdown(f"**Explanation {i+1}**")
            st.markdown(f"Time: {exp.get('timestamp', 'Unknown')}")
            st.markdown(f"Prediction: {exp.get('prediction', 'Unknown')}")

            if exp.get('probability') is not None:
                st.markdown(f"Confidence: {exp.get('probability', 0):.2f}")

            # Extract SHAP values for this explanation
            shap_values = exp.get('shap_values', None)
            if shap_values is not None:
                # Create a simplified bar chart of SHAP values
                feature_vals = {}

                # Handle different formats of SHAP values
                if isinstance(shap_values, list):
                    # For multi-class, use the first class for simplicity
                    vals = shap_values[0][0] if shap_values[0].ndim > 1 else shap_values[0]
                else:
                    vals = shap_values[0]

                # Create a dataframe for plotting
                for j, feat in enumerate(feature_names[:len(vals)]):
                    feature_vals[feat] = vals[j]

                # Sort by absolute magnitude
                shap_df = pd.DataFrame({
                    'Feature': list(feature_vals.keys()),
                    'SHAP Value': list(feature_vals.values())
                })
                shap_df = shap_df.sort_values('SHAP Value', key=abs, ascending=False).head(10)

                # Plot
                fig, ax = plt.subplots(figsize=(4, 5))
                bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'])

                # Color based on value
                for j, bar in enumerate(bars):
                    if shap_df['SHAP Value'].iloc[j] > 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('blue')

                ax.set_xlabel('SHAP Value')
                ax.set_title(f'Top Features for Exp {i+1}')
                st.pyplot(fig)
