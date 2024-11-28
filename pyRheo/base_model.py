# pyRheo/base_model.py

import numpy as np

class BaseModel:
    def __init__(self, model="Maxwell", method="RSS", initial_guesses="bayesian", bounds="default"):
        self.model = model
        self.method = method
        self.initial_guesses = initial_guesses
        self.bounds = bounds
        self.fitted_ = False

    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("Model must be fitted before predicting.")
        raise NotImplementedError("Subclasses should implement this method.")
    
    def print_parameters(self):
        if not self.fitted_:
            raise ValueError("Model must be fitted before printing parameters.")
        raise NotImplementedError("Subclasses should implement this method.")
    
    def calculate_rss(self, y_true, y_pred):
        """
        Calculate the Residual Sum of Squares (RSS).

        Parameters:
        y_true (numpy array): The true values.
        y_pred (numpy array): The predicted values.

        Returns:
        float: The calculated RSS.
        """
        return np.sum(((y_true - y_pred)  / y_true)**2)
