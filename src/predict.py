import pandas as pd

def predict_instance(model, input_dict: dict):
    """
    Predict the class for a single instance (dictionary of features).
    """
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]

def predict_proba(model, input_dict: dict) -> float:
    """
    Predict probability of the predicted class for a single instance.
    """
    df = pd.DataFrame([input_dict])
    return float(model.predict_proba(df).max())
