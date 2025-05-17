import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from ml_explainer import ExplainerFactory

def test_shap_explainer():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    explainer = ExplainerFactory.get_explainer(model, method="shap")
    explanation = explainer.explain(X.iloc[[0]])
    assert explanation is not None

def test_lime_explainer():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    explainer = ExplainerFactory.get_explainer(model, method="lime", training_data=X)
    explanation = explainer.explain(X.iloc[[0]])
    assert explanation is not None
