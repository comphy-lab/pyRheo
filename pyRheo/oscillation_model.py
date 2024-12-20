from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
from .base_model import BaseModel
from .rheo_models.oscillation_models import (
    MaxwellModel, SpringPot, FractionalMaxwellGel, FractionalMaxwellLiquid,
    FractionalMaxwellModel, FractionalKelvinVoigtS, FractionalKelvinVoigtD,
    FractionalKelvinVoigtModel, ZenerModel, FractionalZenerSolidS, FractionalZenerLiquidS,
    FractionalZenerLiquidD, FractionalZenerS
)
import numpy as np
import os
import math
import joblib
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

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
    "FractionalZenerS" : FractionalZenerS
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
}

# Dictionary mapping classifier indices to model names
CLASSIFIER_MODELS = {
    0: "Maxwell",
    1: "SpringPot",
    2: "FractionalMaxwellLiquid",
    3: "FractionalMaxwellGel",
    4: "FractionalMaxwell",
    5: "FractionalKelvinVoigt"
}

class OscillationModel(BaseModel):
    def __init__(self, model="Maxwell", method="RSS", initial_guesses="manual", bounds="auto", minimization_algorithm="Nelder-Mead", num_initial_guesses=64):
        super().__init__(model, method, initial_guesses, bounds)
        if model != "auto" and model not in MODEL_FUNCS:
            raise ValueError(f"Model {model} not recognized.")

        self.model = model
        self.model_func = MODEL_FUNCS.get(model)
        self.minimization_algorithm = minimization_algorithm
        self.custom_bounds = None if bounds == "auto" else bounds
        self.num_initial_guesses = num_initial_guesses
        
        if model == "auto":
            # Load the pretrained models
            current_dir = os.path.dirname(__file__)
            pca_prime_path = os.path.join(current_dir, 'pca_models', 'pca_model_prime.joblib')
            pca_double_prime_path = os.path.join(current_dir, 'pca_models', 'pca_model_prime.joblib')
            mlp_path = os.path.join(current_dir, 'mlp_models', 'mlp_model_oscillation.joblib')

            self.pca_prime = joblib.load(pca_prime_path)
            self.pca_double_prime = joblib.load(pca_double_prime_path)
            self.classifier = joblib.load(mlp_path)

    def _createomegaNumpyLogarithmic(self, start, stop, num):
        return np.logspace(np.log10(start), np.log10(stop), num)

    def _auto_select_model(self, G_prime, G_double_prime, omega):
        def _interpolationToDataPointAmount(X, y, n):
            interpolation_function = interp1d(X, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            new_X = self._createomegaNumpyLogarithmic(X[0], X[-1], n)
            
            # Ensure new_X does not go out of bounds
            new_X = np.clip(new_X, X[0], X[-1])
            
            new_y = interpolation_function(new_X)
            return new_X, new_y

        def _getOscillationPCA(StorageModulus, LossModulus):
            principal_components_prime = self.pca_prime.transform(StorageModulus.reshape(1, -1))
            principal_components_double_prime = self.pca_double_prime.transform(LossModulus.reshape(1, -1))
            principal_components = np.hstack((principal_components_prime, principal_components_prime))
            return principal_components

        StorageModulus = gaussian_filter1d(G_prime, sigma=4.2)
        LossModulus = gaussian_filter1d(G_double_prime, sigma=4.2)
        interpolation_prime = _interpolationToDataPointAmount(omega, StorageModulus, 566) # This is according to the data points in the PCA
        interpolatedStorageModulus = interpolation_prime[1]
        interpolation_double_prime = _interpolationToDataPointAmount(omega, LossModulus, 566) # This is according to the data points in the PCA
        interpolatedLossModulus = interpolation_double_prime[1]
        omega = interpolation_prime[0]
        interpolatedStorageModulus = np.log10(interpolatedStorageModulus)
        interpolatedLossModulus = np.log10(interpolatedLossModulus)
        interpolatedStorageModulus = StandardScaler().fit_transform(interpolatedStorageModulus.reshape(-1, 1)).flatten() # Improved performance as it handles better the specific distribution of the experimental data
        interpolatedLossModulus = StandardScaler().fit_transform(interpolatedLossModulus.reshape(-1, 1)).flatten() # Improved performance as it handles better the specific distribution of the experimental data
        
        pca_components_ = _getOscillationPCA(interpolatedStorageModulus, interpolatedLossModulus)
        prediction = self.classifier.predict(pca_components_)[0]
        predicted_model = CLASSIFIER_MODELS[prediction]
        print(f"Predicted Model: {predicted_model}")
        return predicted_model

    def fit(self, omega, G_prime, G_double_prime, initial_guesses=None):
        if self.model == "auto":
            self.model = self._auto_select_model(G_prime, G_double_prime, omega)
            self.model_func = MODEL_FUNCS[self.model]
        if initial_guesses is None:
            initial_guesses = self._generate_initial_guess(G_prime, G_double_prime, use_log=(self.initial_guesses == "random"))

        if self.initial_guesses == "bayesian" and self.model in ["FractionalMaxwellLiquid", "FractionalMaxwellGel", "FractionalMaxwell"]:
            print(f"Bayesian method not supported for model {self.model}. Switching to random method.")
            self.initial_guesses = "random"

        if self.initial_guesses == "manual":
            return self._fit_model(omega, G_prime, G_double_prime, *initial_guesses, model_func=self.model_func)
        elif self.initial_guesses == "random":
            return self._fit_model_random(omega, G_prime, G_double_prime, model_func=self.model_func)
        elif self.initial_guesses == "bayesian":
            return self._fit_model_bayesian(omega, G_prime, G_double_prime, model_func=self.model_func)

    def _fit_model(self, omega, G_prime, G_double_prime, *initial_guesses, model_func):
        y_true = np.concatenate([G_prime, G_double_prime])

        def residuals(params):
            y_pred = model_func(*params, omega)
            residual = y_true - y_pred
            weights = y_true
            return np.sum((residual / weights)**2)

        bounds = self._get_bounds(initial_guesses, G_prime, G_double_prime, use_log=False)
        print("Using bounds:", bounds)

        result = minimize(residuals, initial_guesses, method=self.minimization_algorithm, bounds=bounds)
        self.params_ = result.x
        y_pred = model_func(*self.params_, omega)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_random(self, omega, G_prime, G_double_prime, model_func):
        y_true = np.concatenate([G_prime, G_double_prime])

        def residuals(params):
            y_pred = model_func(*params, omega)
            residual = y_true - y_pred
            weights = y_true
            return np.sum((residual / weights)**2)

        best_rss = np.inf
        best_params = None
        best_initial_guess = None  # Variable to store the best initial guess

        for _ in range(self.num_initial_guesses):
            initial_guess = self._generate_initial_guess(G_prime, G_double_prime, use_log=False)
            bounds = self._get_bounds(initial_guess, G_prime, G_double_prime, use_log=False)
            result = minimize(residuals, initial_guess, method=self.minimization_algorithm, bounds=bounds)
            if result.success and result.fun < best_rss:
                best_rss = result.fun
                best_params = result.x
                best_initial_guess = initial_guess  # Update the best initial guess

        self.params_ = best_params
        if best_params is None:
            print("Optimization failed to find a solution.")
        else:
            print("Best initial guess was:", best_initial_guess)

        y_pred = model_func(*self.params_, omega)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_bayesian(self, omega, G_prime, G_double_prime, model_func):
        y_true = np.concatenate([G_prime, G_double_prime])

        def residuals(log_params):
            params = [10 ** param if name not in ['alpha', 'beta'] else param for param, name in zip(log_params, MODEL_PARAMS[self.model])]
            y_pred = model_func(*params, omega)
            residual = y_true - y_pred
            weights = y_true
            normalized_residuals = residual / y_true
            rss = np.sum((normalized_residuals)**2)
            #print(rss)
            return rss

        search_space = self._get_search_space(G_prime, G_double_prime)
        print("Search space:", search_space)

        result = gp_minimize(residuals, search_space, n_calls=self.num_initial_guesses, acq_func="EI", xi=0.01, initial_point_generator="sobol", n_initial_points=self.num_initial_guesses // 2)

        # Getting the best result from gp_minimize
        initial_guess_log = result.x
        print("Best initial guess was:", initial_guess_log)
    
        # Transforming initial guesses back to original scale
        initial_guess = [10 ** param if name not in ['alpha', 'beta'] else param for param, name in zip(result.x, MODEL_PARAMS[self.model])]

        # Get bounds in the original scale
        bounds = self._get_bounds(initial_guess, G_prime, G_double_prime, use_log=False)

        # Residuals function in the original parameter space
        def residuals_original_scale(params):
            log_params = [np.log10(param) if name not in ['alpha', 'beta'] else param for param, name in zip(params, MODEL_PARAMS[self.model])]
            return residuals(log_params)

        # Use minimize with the initial guesses from gp_minimize
        result_minimize = minimize(residuals_original_scale, initial_guess, method=self.minimization_algorithm, bounds=bounds)
    
        # Update parameters with the results from minimize
        self.params_ = result_minimize.x

        # Predict and calculate RSS
        y_pred = model_func(*self.params_, omega)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        # Update fitted state
        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _generate_initial_guess(self, G_prime, G_double_prime, use_log):
        initial_guess = []
        alpha = None

        for name in MODEL_PARAMS[self.model]:
            if name == 'alpha':
                alpha = np.random.uniform(0, 1)
                initial_guess.append(alpha)
            elif name == 'beta':
                if alpha is not None:
                    beta = np.random.uniform(0, alpha)
                else:
                    beta = np.random.uniform(0, 1)
                initial_guess.append(beta)
            else:
                range_min, range_max = self._get_param_bounds(G_prime, G_double_prime)
                initial_guess.append(np.random.uniform(np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return initial_guess

    def _get_bounds(self, initial_guess, G_prime, G_double_prime, use_log):
        if self.custom_bounds:
            return self.custom_bounds

        bounds = []
        alpha_bound = None

        for name in MODEL_PARAMS[self.model]:
            if name == 'alpha':
                alpha_bound = (0, 1)
                bounds.append(alpha_bound)
            elif name == 'beta':
                if alpha_bound is not None:
                    beta_bound = (0, initial_guess[MODEL_PARAMS[self.model].index('alpha')])
                else:
                    beta_bound = (0, 1)
                bounds.append(beta_bound)
            else:
                range_min, range_max = self._get_param_bounds(G_prime, G_double_prime)
                bounds.append((np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return bounds

    def _get_param_bounds(self, G_prime, G_double_prime):
        range_min = np.min(G_double_prime) / 100
        range_max = np.max(G_prime) * 100
        return (range_min, range_max)

    def _get_search_space(self, G_prime, G_double_prime):
        search_space = []
        alpha_bound = Real(0, 1)

        for name in MODEL_PARAMS[self.model]:
            if name == 'alpha':
                search_space.append(alpha_bound)
            elif name == 'beta':
                search_space.append(Real(0, 1))  # Initial dummy bound for search space
            else:
                range_min, range_max = self._get_param_bounds(G_prime, G_double_prime)
                search_space.append(Real(np.log10(range_min), np.log10(range_max)))  # Log10 search space

        return search_space

    def predict(self, omega):
        if not self.fitted_:
            raise ValueError("Model must be fitted before predicting.")
        return self._predict_model(omega, self.model_func)

    def _predict_model(self, omega, model_func):
        y_pred = model_func(*self.params_, omega)
        half = len(y_pred) // 2
        G_prime = y_pred[:half]
        G_double_prime = y_pred[half:]
        return G_prime, G_double_prime

    def print_parameters(self):
        if not self.fitted_:
            raise ValueError("Model must be fitted before printing parameters.")
        param_names = MODEL_PARAMS[self.model]
        for name, param in zip(param_names, self.params_):
            print(f"{name}: {param}")
        print(f"RSS: {self.rss_}")
        
    def get_parameters(self):
        if not self.fitted_:
            raise ValueError("Model must be fitted before retrieving parameters.")

        param_names = MODEL_PARAMS[self.model]
        parameters = {name: param for name, param in zip(param_names, self.params_)}
        parameters["RSS"] = self.rss_
        return parameters

    def print_error(self):
        if not hasattr(self, 'y_true') or not hasattr(self, 'y_pred'):
            raise ValueError("Model must be fitted before calculating the error.")

        absolute_error = np.abs(self.y_true - self.y_pred)
        percentage_error = (absolute_error / self.y_true) * 100
        mean_percentage_error = np.mean(percentage_error)

        print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")

    def plot(self, omega, G_prime, G_double_prime, dpi=1200, savefig=False, filename="plot.png", file_format="png"):
        if not self.fitted_:
            raise ValueError("Model must be fitted before plotting.")
    
        import matplotlib.pyplot as plt
        #import scienceplots
        #plt.style.use(['science', 'nature', "bright"])

        # Predict G_prime, G_double_prime using the fitted model
        G_prime_pred, G_double_prime_pred = self.predict(omega)

        # Plot the results
        plt.figure(figsize=(3.2, 3))
        plt.plot(omega, G_prime, 'o', markersize=6, label=r'$G^{\prime}(\omega)$')
        plt.plot(omega, G_double_prime, 'o', fillstyle='none', markersize=6, label=r'$G^{\prime \prime}(\omega)$')
        plt.plot(omega, G_prime_pred, '--', lw=2, color='k', label='fit')
        plt.plot(omega, G_double_prime_pred, '--', color='k', lw=2)
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$\omega$ [rad s$^{-1}$]', fontsize=14)
        plt.ylabel(r'$G^{\prime}(\omega), G^{\prime \prime}(\omega)$ [Pa]', fontsize=14)
        plt.legend(fontsize=13.5)
        plt.grid(False)
        plt.tight_layout()

        if savefig:
            plt.savefig(filename, dpi=dpi, format=file_format, bbox_inches='tight')

        plt.show()
