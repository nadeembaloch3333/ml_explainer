from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer

class ExplainerFactory:
    """
    Factory class to create explainers based on the chosen method.
    """

    @staticmethod
    def get_explainer(model, method="shap", training_data=None, mode="classification"):
        method = method.lower()
        if method == "shap":
            return SHAPExplainer(model)
        elif method == "lime":
            if training_data is None:
                raise ValueError("LIME requires training_data to initialize the explainer.")
            return LIMEExplainer(model, training_data=training_data, mode=mode)
        else:
            raise ValueError(f"Unknown explanation method: {method}")