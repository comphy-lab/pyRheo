from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
from .base_model import BaseModel
from .rheo_models.creep_models import (
    MaxwellModel, SpringPot, FractionalMaxwellGel, FractionalMaxwellLiquid,
    FractionalMaxwellModel, FractionalKelvinVoigtS, FractionalKelvinVoigtD,
    FractionalKelvinVoigtModel, ZenerModel, FractionalZenerSolidS, FractionalZenerLiquidS,
    FractionalZenerLiquidD, FractionalZenerS
)
import numpy as np
import math
import joblib
import os
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

CLASSIFIER_MODELS = {
    0: "Maxwell",
    1: "SpringPot",
    #2: "FractionalMaxwellLiquid",
    #3: "FractionalMaxwellGel",
    2: "FractionalMaxwell",
    3: "FractionalKelvinVoigt"
}

class CreepModel(BaseModel):
    def __init__(self, model="Maxwell", method="RSS", initial_guesses="manual", bounds="auto", minimization_algorithm="Powell", num_initial_guesses=64, mittag_leffler_type="Pade32"):
        super().__init__(model, method, initial_guesses, bounds)
        if model != "auto" and model not in MODEL_FUNCS:
            raise ValueError(f"Model {model} not recognized.")

        self.model = model
        self.model_func = MODEL_FUNCS.get(model)
        self.minimization_algorithm = minimization_algorithm
        self.custom_bounds = None if bounds == "auto" else bounds
        self.num_initial_guesses = num_initial_guesses
        self.mittag_leffler_type = mittag_leffler_type

        # Load pretrained models if the model is set to "auto"
        if model == "auto":
            # Load the pretrained models
            current_dir = os.path.dirname(__file__)
            pca_path = os.path.join(current_dir, 'pca_models', 'pca_model_creep.joblib')
            mlp_path = os.path.join(current_dir, 'mlp_models', 'mlp_model_creep.joblib')

            self.pca = joblib.load(pca_path)
            self.classifier = joblib.load(mlp_path)

    def _createTimeNumpyLogarithmic(self, start, stop, num):
        return np.logspace(np.log10(start), np.log10(stop), num)

    def _auto_select_model(self, J_creep, time):
        def _interpolationToDataPointAmount(X, y, n):
            interpolation_function = interp1d(X, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            new_X = self._createTimeNumpyLogarithmic(X[0], X[-1], n)
            
            # Ensure new_X does not go out of bounds
            new_X = np.clip(new_X, X[0], X[-1])
            
            new_y = interpolation_function(new_X)
            return new_X, new_y

        def _integration(series, time):
            timeDiff = np.diff(time)
            result = np.sum(np.multiply(series[:-1], timeDiff))
            return result

        def _getRelaxtionPCA(creepModulus, timeValues):
            principal_components = self.pca.transform(creepModulus.reshape(1, -1))
            return principal_components

        creepModulus = gaussian_filter1d(J_creep, sigma=4.2)
        interpolation = _interpolationToDataPointAmount(time, creepModulus, 160) # This is according to the data points in the PCA
        interpolatedcreepModulus = interpolation[1]
        time = interpolation[0]
        interpolatedcreepModulus = np.log10(interpolatedcreepModulus) # Improved performance as it handles better the specific distribution of the experimental data
        interpolatedcreepModulus = StandardScaler().fit_transform(interpolatedcreepModulus.reshape(-1, 1)).flatten() # Improved performance as it handles better the specific distribution of the experimental data
        #integral = _integration(interpolatedcreepModulus, time)
        
        pca_components = _getRelaxtionPCA(interpolatedcreepModulus, time)
        #components = np.hstack((pca_components, np.array(integral).reshape(-1, 1)))
        prediction = self.classifier.predict(pca_components)[0]
        #prediction = self.classifier.predict(pca_components)[0] ############ This can change
        predicted_model = CLASSIFIER_MODELS[prediction]
        print(f"Predicted Model: {predicted_model}")
        return predicted_model

    def fit(self, time, J_creep, initial_guesses=None):
        if self.model == "auto":
            self.model = self._auto_select_model(J_creep, time)
            self.model_func = MODEL_FUNCS[self.model]
        if initial_guesses is None:
            initial_guesses = self._generate_initial_guess(J_creep, use_log=(self.initial_guesses == "random"))

        #if self.initial_guesses == "bayesian" and self.model in ["FractionalMaxwellLiquid", "FractionalMaxwellGel", "FractionalMaxwellModel"]:
        #    print(f"Bayesian method not supported for model {self.model}. Switching to random method.")
        #    self.initial_guesses = "random"

        if self.initial_guesses == "manual":
            return self._fit_model(time, J_creep, *initial_guesses, model_func=self.model_func)
        elif self.initial_guesses == "random":
            return self._fit_model_random(time, J_creep, model_func=self.model_func)
        elif self.initial_guesses == "bayesian":
            return self._fit_model_bayesian(time, J_creep, model_func=self.model_func)

    def _fit_model(self, time, J_creep, *initial_guesses, model_func):
        y_true = J_creep

        def residuals(params):
            if 'mittag_leffler_type' in model_func.__code__.co_varnames:
                y_pred = model_func(*params, time, mittag_leffler_type=self.mittag_leffler_type)
            else:
                y_pred = model_func(*params, time)
            residual = y_true - y_pred
            weights = y_true
            return np.sum((residual / weights)**2)

        bounds = self._get_bounds(initial_guesses, J_creep, use_log=False)
        print("Using bounds:", bounds)

        result = minimize(residuals, initial_guesses, method=self.minimization_algorithm, bounds=bounds)
        self.params_ = result.x
        if 'mittag_leffler_type' in model_func.__code__.co_varnames:
            y_pred = model_func(*self.params_, time, mittag_leffler_type=self.mittag_leffler_type)
        else:
            y_pred = model_func(*self.params_, time)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_random(self, time, J_creep, model_func):
        y_true = J_creep

        def residuals(params):
            if 'mittag_leffler_type' in model_func.__code__.co_varnames:
                y_pred = model_func(*params, time, mittag_leffler_type=self.mittag_leffler_type)
            else:
                y_pred = model_func(*params, time)
            residual = y_true - y_pred
            weights = y_true
            return np.sum((residual / weights)**2)

        best_rss = np.inf
        best_params = None
        best_initial_guess = None  # Variable to store the best initial guess

        for _ in range(self.num_initial_guesses):
            try:
                initial_guess = self._generate_initial_guess(J_creep, use_log=False)
                bounds = self._get_bounds(initial_guess, J_creep, use_log=False)
                result = minimize(residuals, initial_guess, method=self.minimization_algorithm, bounds=bounds)
                if result.success and result.fun < best_rss:
                    best_rss = result.fun
                    best_params = result.x
                    best_initial_guess = initial_guess  # Update the best initial guess
            except Exception as e:
                print(f"Attempt failed with error: {e}")
                continue

        self.params_ = best_params
        if best_params is None:
            print("Optimization failed to find a solution.")
        else:
            print("Best initial guess was:", best_initial_guess)

        if 'mittag_leffler_type' in model_func.__code__.co_varnames:
            y_pred = model_func(*self.params_, time, mittag_leffler_type=self.mittag_leffler_type)
        else:
            y_pred = model_func(*self.params_, time)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_bayesian(self, time, J_creep, model_func):
        y_true = J_creep

        def residuals(log_params):
            params = [10 ** param if name not in ['alpha', 'beta'] else param for param, name in zip(log_params, MODEL_PARAMS[self.model])]
            if 'mittag_leffler_type' in model_func.__code__.co_varnames:
                y_pred = model_func(*params, time, mittag_leffler_type=self.mittag_leffler_type)
            else:
                y_pred = model_func(*params, time)
            residual = y_true - y_pred
            weights = y_true
            normalized_residuals = residual / y_true
            return np.sum(np.abs(normalized_residuals))


        search_space = self._get_search_space(J_creep)
        print("Search space:", search_space)

        result = gp_minimize(residuals, search_space, n_calls=self.num_initial_guesses, acq_func="EI", xi=0.01, initial_point_generator="sobol", n_initial_points=self.num_initial_guesses // 2)

        # Getting the best result from gp_minimize
        initial_guess_log = result.x
        print("Best initial guess was:", initial_guess_log)
    
        # Transforming initial guesses back to original scale
        initial_guess = [10 ** param if name not in ['alpha', 'beta'] else param for param, name in zip(result.x, MODEL_PARAMS[self.model])]

        # Get bounds in the original scale
        bounds = self._get_bounds(initial_guess, J_creep, use_log=False)

        # Residuals function in the original parameter space
        def residuals_original_scale(params):
            log_params = [np.log10(param) if name not in ['alpha', 'beta'] else param for param, name in zip(params, MODEL_PARAMS[self.model])]
            return residuals(log_params)

        # Use minimize with the initial guesses from gp_minimize
        result_minimize = minimize(residuals_original_scale, initial_guess, method=self.minimization_algorithm, bounds=bounds)
    
        # Update parameters with the results from minimize
        self.params_ = result_minimize.x

        # Predict and calculate RSS
        if 'mittag_leffler_type' in model_func.__code__.co_varnames:
            y_pred = model_func(*self.params_, time, mittag_leffler_type=self.mittag_leffler_type)
        else:
            y_pred = model_func(*self.params_, time)
        self.rss_ = self.calculate_rss(y_true, y_pred)

        # Update fitted state
        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _generate_initial_guess(self, J_creep, use_log):
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
                range_min, range_max = self._get_param_bounds(J_creep)
                initial_guess.append(np.random.uniform(np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return initial_guess

    def _get_bounds(self, initial_guess, J_creep, use_log):
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
                range_min, range_max = self._get_param_bounds(J_creep)
                bounds.append((np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return bounds

    def _get_param_bounds(self, J_creep):
        range_min = np.min(1/J_creep) / 10
        range_max = np.max(1/J_creep) * 10
        return (range_min, range_max)

    def _get_search_space(self, J_creep):
        search_space = []
        alpha_bound = Real(0, 1)

        for name in MODEL_PARAMS[self.model]:
            if name == 'alpha':
                search_space.append(alpha_bound)
            elif name == 'beta':
                search_space.append(Real(0, 1))  # Initial dummy bound for search space
            else:
                range_min, range_max = self._get_param_bounds(J_creep)
                search_space.append(Real(np.log10(range_min), np.log10(range_max)))  # Log10 search space

        return search_space

    def predict(self, time):
        if not self.fitted_:
            raise ValueError("Model must be fitted before predicting.")
        return self._predict_model(time, self.model_func)

    def _predict_model(self, time, model_func):
        y_pred = model_func(*self.params_, time)
        J_creep = y_pred
        return J_creep

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

    def plot(self, time, J_creep, savefig=False, dpi=1200, filename="plot.png", file_format="png"):
        if not self.fitted_:
            raise ValueError("Model must be fitted before plotting.")
    
        import matplotlib.pyplot as plt
        #import scienceplots
        #plt.style.use(['science', 'nature', "bright"])

        # Predict J_creep using the fitted model
        J_creep_pred = self.predict(time)

        # Plot the results
        plt.figure(figsize=(3.2, 3))
        plt.plot(time, J_creep, 'o', markersize=6, label=r'$J(t)$')
        plt.plot(time, J_creep_pred, '--', color='k', lw=2, label='fit')
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        #plt.xticks([], [])
        #plt.yticks([], [])
        plt.xlabel(r'$t$ [s]', fontsize=16)
        plt.ylabel(r'$J(t)$ [Pa$^{-1}$]', fontsize=16)
        plt.legend(fontsize=13, loc='upper left')
        plt.grid(False)
        plt.tight_layout()

        if savefig:
            plt.savefig(filename, dpi=dpi, format=file_format, bbox_inches='tight')

        plt.show()
