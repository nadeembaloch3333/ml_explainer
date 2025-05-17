from lime.lime_tabular import LimeTabularExplainer
from .base_explainer import BaseExplainer
import numpy as np

class LIMEExplainer(BaseExplainer):
    """
    Explainer that uses LIME to explain model predictions.
    """

    def __init__(self, model, training_data=None, mode="classification"):
        super().__init__(model)
        if training_data is None:
            raise ValueError("LIMEExplainer requires training_data for initialization.")
        self.training_data = training_data
        self.mode = mode
        self.explainer = LimeTabularExplainer(
            training_data=np.array(training_data),
            mode=mode,
            feature_names=training_data.columns if hasattr(training_data, "columns") else None,
            class_names=None
        )

    def explain(self, input_data, sample_idx=0):
        """
        Returns LIME explanation for the input data.
        sample_idx: which row of input_data to explain (default: 0)
        """
        exp = self.explainer.explain_instance(
            np.array(input_data.iloc[sample_idx]),
            self.model.predict_proba if self.mode == "classification" else self.model.predict,
            num_features=len(input_data.columns)
        )
        return exp

    def visualize(self, input_data, sample_idx=0):
        """
        Visualizes LIME explanation for the input data.
        """
        exp = self.explain(input_data, sample_idx=sample_idx)
        from IPython.display import display, HTML
        display(HTML(exp.as_html()))