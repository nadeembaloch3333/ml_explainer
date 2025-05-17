from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    """
    Abstract base class for all explainers.
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def explain(self, input_data):
        """
        Generate explanation for the given input data.
        """
        pass

    @abstractmethod
    def visualize(self, input_data):
        """
        Visualize the explanation for the given input data.
        """
        pass
