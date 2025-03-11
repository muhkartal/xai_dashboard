# XAI Dashboard

![XAI Dashboard](https://img.shields.io/badge/XAI-Dashboard-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.19.0%2B-red)
![SHAP](https://img.shields.io/badge/SHAP-0.42.0%2B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive dashboard for Explainable AI (XAI) that helps users understand machine learning models through SHAP (SHapley Additive exPlanations) values.

## 🌟 Features

-  **Model Integration**

   -  Upload pre-trained models (.pkl or .joblib)
   -  Train models directly in the app
   -  Automatic model type detection

-  **Data Handling**

   -  Upload datasets for analysis
   -  Feature importance visualization
   -  Dataset exploration and statistics

-  **Explainability**

   -  Local explanations for individual predictions
   -  Global explanations for overall model behavior
   -  Multiple visualization types (bar plots, force plots, beeswarm plots)

-  **Advanced Analysis**

   -  Compare multiple explanations side by side
   -  Detailed model metadata inspection
   -  Cross-validation performance metrics

-  **Export Options**
   -  Save visualizations as PNG images
   -  Export SHAP values as CSV
   -  Download interactive HTML reports

## 📋 Requirements

-  Python 3.8 or higher
-  Dependencies listed in `requirements.txt`

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/xai-dashboard.git
cd xai-dashboard
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

## 🔧 Usage

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Access the dashboard in your browser at `http://localhost:8501`

3. Use the sidebar navigation to:
   -  Upload or train models
   -  Generate explanations
   -  Compare results
   -  Export visualizations

## 📊 Dashboard Pages

### Home

Upload models and datasets, view model summaries and basic dataset statistics.

### Train Model

Train machine learning models directly in the app:

-  Select features and target variables
-  Choose model types (Random Forest, XGBoost, etc.)
-  Configure hyperparameters
-  Evaluate with cross-validation
-  Save trained models for immediate use

### Local Explanation

Get explanations for individual predictions:

-  Upload or create test instances
-  Visualize SHAP values with multiple plot types
-  Export explanations in various formats

### Global Explanation

Understand overall model behavior:

-  Generate dataset-level SHAP explanations
-  Visualize feature importance
-  Explore feature dependencies

### Comparison View

Compare multiple explanations side by side to track changes across different instances or models.

### Model Summary

Detailed model inspection:

-  View model architecture and hyperparameters
-  Analyze feature importance
-  Run performance evaluations

## 🏗️ Project Structure

```
xai_dashboard/
├── app.py                   # Main Streamlit application
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── model_utils.py       # Model loading and metadata
│   ├── data_utils.py        # Data processing and analysis
│   ├── shap_utils.py        # SHAP explanations
│   └── ui_utils.py          # UI components
├── requirements.txt         # Dependencies
└── README.md                # Documentation
```

## 🔍 Supported Models

The dashboard supports a variety of machine learning models:

-  **Classification Models**

   -  Random Forest
   -  Gradient Boosting
   -  Logistic Regression
   -  Support Vector Machines
   -  XGBoost

-  **Regression Models**
   -  Random Forest
   -  Gradient Boosting
   -  Linear Regression
   -  Support Vector Regression
   -  XGBoost

## 🛠️ Technical Implementation

-  **Framework**: Streamlit for the web interface
-  **Explainability**: SHAP library for model interpretability
-  **Visualization**: Matplotlib, Seaborn for static plots
-  **ML Backend**: Scikit-learn and XGBoost for model training and evaluation
-  **Processing**: Pandas and NumPy for data manipulation

## 📚 How SHAP Works

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

In this dashboard:

1. **TreeExplainer** is used for tree-based models (faster computation)
2. **KernelExplainer** is used as a fallback for other model types
3. Both local (individual prediction) and global (entire dataset) explanations are provided

## 📊 Sample Visualizations

The dashboard produces various visualization types:

-  **SHAP Summary Plots**: Show feature importance and impact direction
-  **SHAP Force Plots**: Illustrate how each feature contributes to a specific prediction
-  **Feature Importance Plots**: Display overall importance of features to the model
-  **Comparison Views**: Side-by-side examination of different explanations
-  **Performance Metrics**: Visualizations of model accuracy, confusion matrices, etc.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Muhammed Kartal - [kartal.dev](https://kartal.dev)

## 🙏 Acknowledgements

-  [SHAP](https://github.com/slundberg/shap) library for model explanations
-  [Streamlit](https://streamlit.io/) for the interactive web interface
-  [Scikit-learn](https://scikit-learn.org/) for machine learning components
