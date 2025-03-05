from .rheo_models.oscillation_models import (
    MaxwellModel, SpringPot, FractionalMaxwellGel, FractionalMaxwellLiquid,
    FractionalMaxwellModel, FractionalKelvinVoigtS, FractionalKelvinVoigtD,
    FractionalKelvinVoigtModel, ZenerModel, FractionalZenerSolidS, FractionalZenerLiquidS,
    FractionalZenerLiquidD, FractionalZenerS, FractionalZener
)
import numpy as np
import os
import math
import warnings


# Dummy BaseModel class for completeness (this should be defined elsewhere in your project)
class BaseModel:
    def __init__(self, model, method, initial_guesses, bounds):
        pass


# Dictionary mapping model names to their respective classes
MODEL_FUNCS = {
    "Maxwell": MaxwellModel,
    "SpringPot": SpringPot,
    "FractionalMaxwellGel": FractionalMaxwellGel,
    "FractionalMaxwellLiquid": FractionalMaxwellLiquid,
    "FractionalMaxwell": FractionalMaxwellModel,
    "FractionalKelvinVoigtS": FractionalKelvinVoigtS,
    "FractionalKelvinVoigtD": FractionalKelvinVoigtD,
    "FractionalKelvinVoigt": FractionalKelvinVoigtModel,
    "Zener": ZenerModel,
    "FractionalZenerSolidS": FractionalZenerSolidS,
    "FractionalZenerLiquidS": FractionalZenerLiquidS,
    "FractionalZenerLiquidD": FractionalZenerLiquidD,
    "FractionalZenerS" : FractionalZenerS,
    "FractionalZener" : FractionalZener
}

# Dictionary mapping model names to their respective parameters
MODEL_PARAMS = {
    "Maxwell": ["G_s", "eta_s"],
    "SpringPot": ["V", "alpha"],
    "FractionalMaxwellGel": ["G_s", "V", "alpha"],
    "FractionalMaxwellLiquid": ["G", "eta_s", "beta"],
    "FractionalMaxwell": ["G", "V", "alpha", "beta"],
    "FractionalKelvinVoigtS": ["G_p", "V", "alpha"],
    "FractionalKelvinVoigtD": ["G", "eta_p", "beta"],
    "FractionalKelvinVoigt": ["G", "V", "alpha", "beta"],
    "Zener": ["G_p", "G_s", "eta_s"],
    "FractionalZenerSolidS": ["G_p", "G_s", "V", "alpha"],
    "FractionalZenerLiquidS": ["G_p", "G", "eta_s", "beta"],
    "FractionalZenerLiquidD": ["eta_p", "G", "eta_s", "beta"],
    "FractionalZenerS": ["G_p", "G", "V", "alpha", "beta"],
    "FractionalZener": ["G", "V", "K", "alpha", "beta", "kappa"]
}


# New class to evaluate the model given fixed parameters
class SAOSEvaluator:
    def __init__(self, model="Maxwell"):
        if model not in MODEL_FUNCS:
            raise ValueError(f"Model {model} not recognized.")
        self.model = model
        self.model_func = MODEL_FUNCS[model]
    
    def compute_model(self, params, omega):
        if len(params) != len(MODEL_PARAMS[self.model]):
            raise ValueError(f"Incorrect number of parameters for model {self.model}. Expected {len(MODEL_PARAMS[self.model])}, got {len(params)}.")
        
        model_values = self.model_func(*params, omega)
        half = len(model_values) // 2
        G_prime = model_values[:half]
        G_double_prime = model_values[half:]
        
        return G_prime, G_double_prime
        
        
# Now define the OscillationModel subclass that issues a deprecation warning when used.
class OscillationEvaluator(SAOSEvaluator):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "OscillationEvaluator will be deprecated and will be removed in future versions. Please use SAOSEvaluator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
