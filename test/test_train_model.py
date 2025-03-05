import pytest
from sklearn.ensemble import RandomForestClassifier
from src.train_model import train_model, evaluate_model

def test_train_model():
    X_train = [[1,2],[3,4],[5,6]]
    y_train = [0,1,0]
    model = train_model(X_train, y_train, n_estimators=5)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 5

def test_evaluate_model():
    model = RandomForestClassifier(random_state=42)
    X_train = [[1,2],[3,4],[5,6]]
    y_train = [0,1,0]
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, [[1,2],[3,4]], [0,1])
    assert 'accuracy' in metrics
    assert 0.0 <= metrics['accuracy'] <= 1.0
