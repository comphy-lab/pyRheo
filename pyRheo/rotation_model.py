from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
from .base_model import BaseModel
from .rheo_models.rotation_models import (
    HerschelBulkley, Bingham, PowerLaw, CarreauYasuda, Cross, Casson
)
import numpy as np
import math
import joblib
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

MODEL_FUNCS = {
    "HerschelBulkley": HerschelBulkley,
    "Bingham": Bingham,
    "PowerLaw": PowerLaw,
    "CarreauYasuda": CarreauYasuda,
    "Cross": Cross,
    "Casson": Casson
}

MODEL_PARAMS = {
    "HerschelBulkley": ["sigma_y", "k", "n"],
    "Bingham": ["sigma_y", "eta_p"],
    "PowerLaw": ["k", "n"],
    "CarreauYasuda": ["eta_inf", "eta_zero", "k", "a", "n"],
    "Cross": ["eta_inf", "eta_zero", "k", "a", "n"],
    "Casson": ["sigma_y", "eta_p"]
}

CLASSIFIER_MODELS = {
    0: "HerschelBulkley",
    1: "Bingham",
    2: "PowerLaw",
    3: "CarreauYasuda",
    4: "Cross",
    5: "Casson"
}

class RotationModel(BaseModel):
    def __init__(self, model="HerschelBulkley", method="RSS", initial_guesses="manual", bounds="auto", minimization_algorithm="Nelder-Mead", num_initial_guesses=64):
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
            self.pca = joblib.load('pca_models/pca_model_rotation.joblib')
            self.classifier = joblib.load('mlp_models/mlp_model_rotation.joblib')

    def _creategamma_dotNumpyLogarithmic(self, start, stop, num):
        return np.logspace(np.log10(start), np.log10(stop), num)

    def _auto_select_model(self, eta, gamma_dot):
        def _interpolationToDataPointAmount(X, y, n):
            interpolation_function = interp1d(X, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            new_X = self._creategamma_dotNumpyLogarithmic(X[0], X[-1], n)
            
            # Ensure new_X does not go out of bounds
            new_X = np.clip(new_X, X[0], X[-1])
            
            new_y = interpolation_function(new_X)
            return new_X, new_y

        def _integration(series, gamma_dot):
            gamma_dotDiff = np.diff(gamma_dot)
            result = np.sum(np.multiply(series[:-1], gamma_dotDiff))
            return result

        def _getRelaxtionPCA(viscosity, gamma_dotValues):
            principal_components = self.pca.transform(viscosity.reshape(1, -1))
            return principal_components

        viscosity = gaussian_filter1d(eta, sigma=4.2)
        interpolation = _interpolationToDataPointAmount(gamma_dot, viscosity, 160) # This is according to the data points in the PCA
        interpolatedviscosity = interpolation[1]
        gamma_dot = interpolation[0]
        interpolatedviscosity = np.log10(interpolatedviscosity) # Improved performance as it handles better the specific distribution of the experimental data
        interpolatedviscosity = StandardScaler().fit_transform(interpolatedviscosity.reshape(-1, 1)).flatten() # Improved performance as it handles better the specific distribution of the experimental data
        #integral = _integration(interpolatedviscosity, gamma_dot)
        
        pca_components = _getRelaxtionPCA(interpolatedviscosity, gamma_dot)
        #components = np.hstack((pca_components, np.array(integral).reshape(-1, 1)))
        prediction = self.classifier.predict(pca_components)[0]
        #prediction = self.classifier.predict(pca_components)[0] ############ This can change
        predicted_model = CLASSIFIER_MODELS[prediction]
        print(f"Predicted Model: {predicted_model}")
        return predicted_model

    def fit(self, gamma_dot, eta, initial_guesses=None):
        if self.model == "auto":
            self.model = self._auto_select_model(eta, gamma_dot)
            self.model_func = MODEL_FUNCS[self.model]
        if initial_guesses is None:
            initial_guesses = self._generate_initial_guess(eta, use_log=(self.initial_guesses == "random"))

        if self.initial_guesses == "bayesian" and self.model in ["FractionalMaxwellLiquid", "FractionalMaxwellGel", "FractionalMaxwellModel"]:
            print(f"Bayesian method not supported for model {self.model}. Switching to random method.")
            self.initial_guesses = "random"

        if self.initial_guesses == "manual":
            return self._fit_model(gamma_dot, eta, *initial_guesses, model_func=self.model_func)
        elif self.initial_guesses == "random":
            return self._fit_model_random(gamma_dot, eta, model_func=self.model_func)
        elif self.initial_guesses == "bayesian":
            return self._fit_model_bayesian(gamma_dot, eta, model_func=self.model_func)

    def _fit_model(self, gamma_dot, eta, *initial_guesses, model_func):
        y_true = eta

        def residuals(params):
            y_pred = model_func(*params, gamma_dot)
            residual = y_true - y_pred
            weights = y_true
            return np.sum((residual / weights)**2)

        bounds = self._get_bounds(initial_guesses, eta, use_log=False)
        print("Using bounds:", bounds)

        result = minimize(residuals, initial_guesses, method=self.minimization_algorithm, bounds=bounds)
        self.params_ = result.x
        y_pred = model_func(*self.params_, gamma_dot)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_random(self, gamma_dot, eta, model_func):
        y_true = eta

        def residuals(params):
            y_pred = model_func(*params, gamma_dot)
            residual = y_true - y_pred
            weights = y_true
            return np.sum((residual / weights)**2)

        best_rss = np.inf
        best_params = None
        best_initial_guess = None  # Variable to store the best initial guess

        for _ in range(self.num_initial_guesses):
            initial_guess = self._generate_initial_guess(eta, use_log=False)
            bounds = self._get_bounds(initial_guess, eta, use_log=False)
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

        y_pred = model_func(*self.params_, gamma_dot)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_bayesian(self, gamma_dot, eta, model_func):
        y_true = eta

        def residuals(log_params):
            params = [10 ** param if name not in ['a', 'n'] else param for param, name in zip(log_params, MODEL_PARAMS[self.model])]
            y_pred = model_func(*params, gamma_dot)
            residual = y_true - y_pred
            weights = y_true
            normalized_residuals = residual / y_true
            return np.sum(np.abs(normalized_residuals))

        search_space = self._get_search_space(eta)
        print("Search space:", search_space)

        result = gp_minimize(residuals, search_space, n_calls=self.num_initial_guesses, acq_func="EI", xi=0.01, initial_point_generator="sobol", n_initial_points=self.num_initial_guesses // 2)

        # Getting the best result from gp_minimize
        initial_guess_log = result.x
        print("Best initial guess was:", initial_guess_log)
    
        # Transforming initial guesses back to original scale
        initial_guess = [10 ** param if name not in ['a', 'n'] else param for param, name in zip(result.x, MODEL_PARAMS[self.model])]

        # Get bounds in the original scale
        bounds = self._get_bounds(initial_guess, eta, use_log=False)

        # Residuals function in the original parameter space
        def residuals_original_scale(params):
            log_params = [np.log10(param) if name not in ['a', 'n'] else param for param, name in zip(params, MODEL_PARAMS[self.model])]
            return residuals(log_params)

        # Use minimize with the initial guesses from gp_minimize
        result_minimize = minimize(residuals_original_scale, initial_guess, method=self.minimization_algorithm, bounds=bounds)
    
        # Update parameters with the results from minimize
        self.params_ = result_minimize.x

        # Predict and calculate RSS
        y_pred = model_func(*self.params_, gamma_dot)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        # Update fitted state
        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _generate_initial_guess(self, eta, use_log):
        initial_guess = []
        alpha = None

        for name in MODEL_PARAMS[self.model]:
            if name == 'n':
                n = np.random.uniform(0, 1)
                initial_guess.append(n)
            else:
                range_min, range_max = self._get_param_bounds(eta)
                initial_guess.append(np.random.uniform(np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return initial_guess

    def _get_bounds(self, initial_guess, eta, use_log):
        if self.custom_bounds:
            return self.custom_bounds

        bounds = []
        n_bound = None

        for name in MODEL_PARAMS[self.model]:
            if name == 'n':
                n_bound = (0, 1)
                bounds.append(n_bound)
            else:
                range_min, range_max = self._get_param_bounds(eta)
                bounds.append((np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return bounds

    def _get_param_bounds(self, eta):
        range_min = np.min(eta) / 10
        range_max = np.max(eta) * 10
        return (range_min, range_max)

    def _get_search_space(self, eta):
        search_space = []
        n_bound = Real(0, 1)

        for name in MODEL_PARAMS[self.model]:
            if name == 'n':
                search_space.append(n_bound)
            else:
                range_min, range_max = self._get_param_bounds(eta)
                search_space.append(Real(np.log10(range_min), np.log10(range_max)))  # Log10 search space

        return search_space

    def predict(self, gamma_dot):
        if not self.fitted_:
            raise ValueError("Model must be fitted before predicting.")
        return self._predict_model(gamma_dot, self.model_func)

    def _predict_model(self, gamma_dot, model_func):
        y_pred = model_func(*self.params_, gamma_dot)
        eta = y_pred
        return eta

    def print_parameters(self):
        if not self.fitted_:
            raise ValueError("Model must be fitted before printing parameters.")
        param_names = MODEL_PARAMS[self.model]
        for name, param in zip(param_names, self.params_):
            print(f"{name}: {param}")
        print(f"RSS: {self.rss_}")

    def print_error(self):
        if not hasattr(self, 'y_true') or not hasattr(self, 'y_pred'):
            raise ValueError("Model must be fitted before calculating the error.")

        absolute_error = np.abs(self.y_true - self.y_pred)
        percentage_error = (absolute_error / self.y_true) * 100
        mean_percentage_error = np.mean(percentage_error)

        print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")

    def plot(self, gamma_dot, eta, savefig=False, filename="plot.png", dpi=300, file_format="png"):
        if not self.fitted_:
            raise ValueError("Model must be fitted before plotting.")
    
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science', 'nature', "bright"])

        # Predict eta using the fitted model
        eta_pred = self.predict(gamma_dot)

        # Plot the results
        plt.figure(figsize=(3.2, 3))
        plt.plot(gamma_dot, eta, 'o', markersize=6, label=r'$\eta(\dot{\gamma})$')
        plt.plot(gamma_dot, eta_pred, '--', color='k', lw=2, label='fit')
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(r'$\dot{\gamma}$ [s$^{-1}$]', fontsize=14)
        plt.ylabel(r'$\eta(\dot{\gamma})$ [Pa s]', fontsize=14)
        plt.legend(fontsize=13)
        plt.grid(False)
        plt.tight_layout()

        if savefig:
            plt.savefig(filename, dpi=dpi, format=file_format, bbox_inches='tight')

        plt.show()
