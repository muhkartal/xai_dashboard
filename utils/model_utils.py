import io
import logging
import joblib
from typing import Dict, Optional, Any, Tuple
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

def load_user_model(uploaded_file) -> Optional[BaseEstimator]:
    """
    Loads model from uploaded file with improved error handling
    """
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            model = joblib.load(io.BytesIO(bytes_data))
            logger.info("User model loaded successfully from uploaded file.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from user upload: {e}")
            return None
    return None

def detect_model_type(model) -> Tuple[str, Optional[int]]:
    """
    Detects if model is classification or regression, and number of classes
    """
    # Try to determine from model attributes
    if hasattr(model, 'classes_'):
        num_classes = len(model.classes_)
        return 'classification', num_classes
    elif hasattr(model, '_classes'):
        num_classes = len(model._classes)
        return 'classification', num_classes
    elif hasattr(model, 'n_classes_'):
        num_classes = model.n_classes_
        return 'classification', num_classes
    elif hasattr(model, 'objective') and 'reg' in str(model.objective).lower():
        return 'regression', None

    # Default assumption
    return 'classification', 2

def get_model_info(model) -> Dict[str, Any]:
    """
    Extracts metadata about the model
    """
    model_type, num_classes = detect_model_type(model)

    # Get model class name
    model_name = type(model).__name__

    # Try to extract hyperparameters
    try:
        if hasattr(model, 'get_params'):
            params = model.get_params()
        else:
            params = {}
    except:
        params = {}

    # Check for feature importance
    has_feature_importance = hasattr(model, 'feature_importances_')

    return {
        'model_name': model_name,
        'model_type': model_type,
        'num_classes': num_classes,
        'params': params,
        'has_feature_importance': has_feature_importance
    }
