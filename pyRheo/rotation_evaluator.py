from .rheo_models.rotation_models import (
    HerschelBulkley, Bingham, PowerLaw, CarreauYasuda, Cross, Casson
)
import numpy as np
import os
import math

# Dummy BaseModel class for completeness (this should be defined elsewhere in your project)
class BaseModel:
    def __init__(self, model, method, initial_guesses, bounds):
        pass

# Dictionary mapping model names to their respective classes
MODEL_FUNCS = {
    "HerschelBulkley": HerschelBulkley,
    "Bingham": Bingham,
    "PowerLaw": PowerLaw,
    "CarreauYasuda": CarreauYasuda,
    "Cross": Cross,
    "Casson": Casson
}

# Dictionary mapping model names to their respective parameters
MODEL_PARAMS = {
    "HerschelBulkley": ["sigma_y", "k", "n"],
    "Bingham": ["sigma_y", "eta_p"],
    "PowerLaw": ["k", "n"],
    "CarreauYasuda": ["eta_inf", "eta_zero", "k", "a", "n"],
    "Cross": ["eta_inf", "eta_zero", "k", "a", "n"],
    "Casson": ["sigma_y", "eta_p"]
}

# New class to evaluate the model given fixed parameters
class RotationEvaluator:
    def __init__(self, model="PowerLaw"):
        if model not in MODEL_FUNCS:
            raise ValueError(f"Model {model} not recognized.")
        self.model = model
        self.model_func = MODEL_FUNCS[model]
    
    def compute_model(self, params, gamma_dot):
        if len(params) != len(MODEL_PARAMS[self.model]):
            raise ValueError(f"Incorrect number of parameters for model {self.model}. Expected {len(MODEL_PARAMS[self.model])}, got {len(params)}.")
        
        model_values = self.model_func(*params, gamma_dot)
 
        eta = model_values
        
        return eta
