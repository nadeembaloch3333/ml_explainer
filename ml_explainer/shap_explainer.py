import shap
from .base_explainer import BaseExplainer

class SHAPExplainer(BaseExplainer):
    """
    Explainer that uses SHAP to explain model predictions.
    """

    def __init__(self, model):
        super().__init__(model)
        self.explainer = shap.Explainer(model)

    def explain(self, input_data):
        """
        Returns SHAP values for the input data.
        """
        return self.explainer(input_data)

    def visualize(self, input_data, class_idx=0):
        """
        Visualizes SHAP values for the input data.
        For multi-class models, specify class_idx (default is 0).
        """
        shap_values = self.explain(input_data)
        single_expl = shap_values[0]
        # If multi-class, .values will be 2D (n_classes, n_features)
        if hasattr(single_expl, "values") and len(single_expl.values.shape) == 2:
            class_expl = shap.Explanation(
                values=single_expl.values[class_idx],
                base_values=single_expl.base_values[class_idx],
                data=single_expl.data,
                feature_names=single_expl.feature_names
            )
            shap.plots.waterfall(class_expl)
        else:
            # For single-class, just plot directly
            shap.plots.waterfall(single_expl)