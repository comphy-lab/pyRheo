from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
from .base_model import BaseModel
from .rheo_models.relaxation_models import (
    MaxwellModel, SpringPot, FractionalMaxwellGel, FractionalMaxwellLiquid,
    FractionalMaxwellModel, FractionalKelvinVoigtS, FractionalKelvinVoigtD,
    FractionalKelvinVoigtModel, ZenerModel, FractionalZenerSolidS, FractionalZenerLiquidS,
    FractionalZenerLiquidD, FractionalZenerS, FractionalZener
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

# Dictionary mapping classifier indices to model names
CLASSIFIER_MODELS = {
    0: "Maxwell",
    1: "SpringPot",
    2: "FractionalMaxwellLiquid",
    3: "FractionalMaxwellGel",
    4: "FractionalMaxwell",
    5: "FractionalKelvinVoigt"
}

class RelaxationModel(BaseModel):
    def __init__(self, model="Maxwell", cost_function="RSS", initial_guesses="manual", bounds="auto", minimization_algorithm="Powell", num_initial_guesses=64, mittag_leffler_type="Pade32"):
        super().__init__(model, cost_function, initial_guesses, bounds)
        if model != "auto" and model not in MODEL_FUNCS:
            raise ValueError(f"Model {model} not recognized.")

        self.model = model
        self.model_func = MODEL_FUNCS.get(model)
        self.minimization_algorithm = minimization_algorithm
        self.custom_bounds = None if bounds == "auto" else bounds
        self.num_initial_guesses = num_initial_guesses
        self.mittag_leffler_type = mittag_leffler_type
        self.cost_function = cost_function
        self.num_parameters = len(MODEL_PARAMS[model]) if model != "auto" else None  



        # Load pretrained models if the model is set to "auto"
        if model == "auto":
            # Load the pretrained models
            current_dir = os.path.dirname(__file__)
            pca_path = os.path.join(current_dir, 'pca_models', 'pca_model_relaxation.joblib')
            mlp_path = os.path.join(current_dir, 'mlp_models', 'mlp_model_relaxation.joblib')

            self.pca = joblib.load(pca_path)
            self.classifier = joblib.load(mlp_path)

    def _createTimeNumpyLogarithmic(self, start, stop, num):
        # Generate logarithmically spaced time points
        return np.logspace(np.log10(start), np.log10(stop), num)
        
    def _calculate_cost(self, y_true, y_pred):
        num_params = self.num_parameters

        if self.cost_function == "RSS":
            residual = y_true - y_pred
            weights = y_true
            return np.sum((residual / weights)**2)
        elif self.cost_function == "MSE":
            return np.mean((y_true - y_pred) ** 2)
        elif self.cost_function == "MAE":
            return np.mean(np.abs(y_true - y_pred))
        elif self.cost_function == "BIC":
            residual = y_true - y_pred
            weights = y_true
            rss = np.sum((residual / weights)**2)
            return rss + num_params * np.log(len(y_true))
        else:
            raise ValueError(f"Cost function {self.cost_function} not recognized.")


    def _auto_select_model(self, G_relax, time):
        # Function to interpolate data points
        def _interpolationToDataPointAmount(X, y, n):
            interpolation_function = interp1d(X, y, kind='linear', bounds_error=False, fill_value='extrapolate')
            new_X = self._createTimeNumpyLogarithmic(X[0], X[-1], n)
            # Ensure new_X does not go out of bounds
            new_X = np.clip(new_X, X[0], X[-1])
            
            new_y = interpolation_function(new_X)
            return new_X, new_y

        # Function to perform numerical integration
        def _integration(series, time):
            timeDiff = np.diff(time)
            result = np.sum(np.multiply(series[:-1], timeDiff))
            return result

        # Function to transform relaxation modulus data using PCA
        def _getRelaxtionPCA(relaxationModulus, timeValues):
            principal_components = self.pca.transform(relaxationModulus.reshape(1, -1))
            return principal_components

        # Preprocess and interpolate the relaxation modulus data
        relaxationModulus = gaussian_filter1d(G_relax, sigma=4.2)
        interpolation = _interpolationToDataPointAmount(time, relaxationModulus, 160) # This is according to the data points in the PCA
        interpolatedRelaxationModulus = interpolation[1]
        time = interpolation[0]
        interpolatedRelaxationModulus = StandardScaler().fit_transform(interpolatedRelaxationModulus.reshape(-1, 1)).flatten() # Improved performance as it handles better the specific distribution of the experimental data
        integral = _integration(interpolatedRelaxationModulus, time)
        
        # Transform the interpolated data using PCA and predict the model type
        pca_components = _getRelaxtionPCA(interpolatedRelaxationModulus, time)
        components = np.hstack((pca_components, np.array(integral).reshape(-1, 1)))
        prediction = self.classifier.predict(pca_components)[0]
        predicted_model = CLASSIFIER_MODELS[prediction]
        print(f"Predicted Model: {predicted_model}")
        return predicted_model

    def fit(self, time, G_relax, initial_guesses=None):
        if self.model == "auto":
            self.model = self._auto_select_model(G_relax, time)
            self.model_func = MODEL_FUNCS[self.model]
            self.num_parameters = len(MODEL_PARAMS[self.model])  # Update for auto-selected model

        if initial_guesses is None:
            initial_guesses = self._generate_initial_guess(G_relax, use_log=(self.initial_guesses == "random"))

        if self.initial_guesses == "manual":
            fit_result = self._fit_model(time, G_relax, *initial_guesses, model_func=self.model_func)
        elif self.initial_guesses == "random":
            fit_result = self._fit_model_random(time, G_relax, model_func=self.model_func)
        elif self.initial_guesses == "bayesian":
            fit_result = self._fit_model_bayesian(time, G_relax, model_func=self.model_func)
        
        self._check_fit_parameters()
        return fit_result

    def _check_fit_parameters(self):
        if not self.fitted_:
            raise ValueError("Model must be fitted before checking parameters.")
        param_names = MODEL_PARAMS[self.model]
        for name, param in zip(param_names, self.params_):
            if name == 'alpha' and param > 0.9:
                print("Consider replacing the alpha springpot element with a dashpot.")
            if name == 'beta' and param < 0.1:
                print("Consider replacing the beta springpot element with a spring.")
            if name == 'kappa' and param > 0.9:
                print("Consider replacing the kappa springpot element with a dashpot.")
            if name == 'kappa' and param < 0.1:
                print("Consider replacing the kappa springpot element with a spring.")

    def _fit_model(self, time, G_relax, *initial_guesses, model_func):
        y_true = G_relax

        def residuals(params):
            if 'mittag_leffler_type' in model_func.__code__.co_varnames:
                y_pred = model_func(*params, time, mittag_leffler_type=self.mittag_leffler_type)
            else:
                y_pred = model_func(*params, time)
            return self._calculate_cost(y_true, y_pred)

        bounds = self._get_bounds(initial_guesses, G_relax, use_log=False)
        print("Using bounds:", bounds)

        result = minimize(residuals, initial_guesses, method=self.minimization_algorithm, bounds=bounds)
        self.params_ = result.x

        if 'mittag_leffler_type' in model_func.__code__.co_varnames:
            y_pred = model_func(*self.params_, time, mittag_leffler_type=self.mittag_leffler_type)
        else:
            y_pred = model_func(*self.params_, time)
        self.cost_ = self._calculate_cost(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_random(self, time, G_relax, model_func):
        y_true = G_relax

        def residuals(params):
            if 'mittag_leffler_type' in model_func.__code__.co_varnames:
                y_pred = model_func(*params, time, mittag_leffler_type=self.mittag_leffler_type)
            else:
                y_pred = model_func(*params, time)
            return self._calculate_cost(y_true, y_pred)

        best_cost = np.inf
        best_params = None
        best_initial_guess = None

        for _ in range(self.num_initial_guesses):
            try:
                initial_guess = self._generate_initial_guess(G_relax, use_log=False)
                bounds = self._get_bounds(initial_guess, G_relax, use_log=False)
                result = minimize(residuals, initial_guess, method=self.minimization_algorithm, bounds=bounds)
                if result.success and result.fun < best_cost:
                    best_cost = result.fun
                    best_params = result.x
                    best_initial_guess = initial_guess
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
        self.cost_ = self._calculate_cost(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _fit_model_bayesian(self, time, G_relax, model_func):
        y_true = G_relax

        def residuals(log_params):
            params = [10 ** param if name not in ['alpha', 'beta', 'kappa'] else param for param, name in zip(log_params, MODEL_PARAMS[self.model])]
            if 'mittag_leffler_type' in model_func.__code__.co_varnames:
                y_pred = model_func(*params, time, mittag_leffler_type=self.mittag_leffler_type)
            else:
                y_pred = model_func(*params, time)
            return self._calculate_cost(y_true, y_pred)

        search_space = self._get_search_space(G_relax)
        print("Search space:", search_space)

        result = gp_minimize(residuals, search_space, n_calls=self.num_initial_guesses, acq_func="EI", xi=0.01, initial_point_generator="sobol", n_initial_points=self.num_initial_guesses // 2)

        initial_guess_log = result.x
        print("Best initial guess was:", initial_guess_log)

        initial_guess = [10 ** param if name not in ['alpha', 'beta', 'kappa'] else param for param, name in zip(result.x, MODEL_PARAMS[self.model])]
        bounds = self._get_bounds(initial_guess, G_relax, use_log=False)

        def residuals_original_scale(params):
            log_params = [np.log10(param) if name not in ['alpha', 'beta', 'kappa'] else param for param, name in zip(params, MODEL_PARAMS[self.model])]
            if 'mittag_leffler_type' in model_func.__code__.co_varnames:
                y_pred = model_func(*params, time, mittag_leffler_type=self.mittag_leffler_type)
            else:
                y_pred = model_func(*params, time)
            return self._calculate_cost(y_true, y_pred)

        result_minimize = minimize(residuals_original_scale, initial_guess, method=self.minimization_algorithm, bounds=bounds)

        self.params_ = result_minimize.x

        if 'mittag_leffler_type' in model_func.__code__.co_varnames:
            y_pred = model_func(*self.params_, time, mittag_leffler_type=self.mittag_leffler_type)
        else:
            y_pred = model_func(*self.params_, time)
        self.cost_ = self._calculate_cost(y_true, y_pred)

        self.fitted_ = True
        self.y_true = y_true
        self.y_pred = y_pred

    def _generate_initial_guess(self, G_relax, use_log):
        initial_guess = []
        alpha = None

        for name in MODEL_PARAMS[self.model]:
            if name == 'alpha':
                alpha = np.random.uniform(0.0001, 0.99)
                initial_guess.append(alpha)
            elif name == 'beta':
                if alpha is not None:
                    beta = np.random.uniform(0.0001, alpha)
                else:
                    beta = np.random.uniform(0.0001, 0.99)
                initial_guess.append(beta)
            elif name == 'kappa':
                kappa = np.random.uniform(0, 1)
                initial_guess.append(kappa)
            else:
                range_min, range_max = self._get_param_bounds(G_relax)
                initial_guess.append(np.random.uniform(np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return initial_guess

    def _get_bounds(self, initial_guess, G_relax, use_log):
        if self.custom_bounds:
            return self.custom_bounds

        bounds = []
        alpha_bound = None

        for name in MODEL_PARAMS[self.model]:
            if name == 'alpha':
                alpha_bound = (0.0001, 0.99)
                bounds.append(alpha_bound)
            elif name == 'beta':
                if alpha_bound is not None:
                    beta_bound = (0.0001, initial_guess[MODEL_PARAMS[self.model].index('alpha')])
                else:
                    beta_bound = (0.0001, 0.99)
                bounds.append(beta_bound)
            elif name == 'kappa':
                bounds.append((0, 1))
            else:
                range_min, range_max = self._get_param_bounds(G_relax)
                bounds.append((np.log10(range_min) if use_log else range_min, np.log10(range_max) if use_log else range_max))

        return bounds

    def _get_param_bounds(self, G_relax):
        range_min = np.min(G_relax) / 1000
        range_max = np.max(G_relax) * 1000
        return (range_min, range_max)

    def _get_search_space(self, G_relax):
        search_space = []
        alpha_bound = Real(0.0001, 0.99)

        for name in MODEL_PARAMS[self.model]:
            if name == 'alpha':
                search_space.append(alpha_bound)
            elif name == 'beta':
                search_space.append(Real(0.0001, 0.99))  # Initial dummy bound for search space
            elif name == 'kappa':
                search_space.append(Real(0, 1))
            else:
                range_min, range_max = self._get_param_bounds(G_relax)
                search_space.append(Real(np.log10(range_min), np.log10(range_max)))  # Log10 search space

        return search_space

    def predict(self, time):
        if not self.fitted_:
            raise ValueError("Model must be fitted before predicting.")
        return self._predict_model(time, self.model_func)

    def _predict_model(self, time, model_func):
        y_pred = model_func(*self.params_, time)
        G_relax = y_pred
        return G_relax

    def print_parameters(self):
        if not self.fitted_:
            raise ValueError("Model must be fitted before printing parameters.")
        
        param_names = MODEL_PARAMS[self.model]
        for name, param in zip(param_names, self.params_):
            print(f"{name}: {param}")
        
        print(f"Cost ({self.cost_function}): {self.cost_}")

    def get_parameters(self):
        if not self.fitted_:
            raise ValueError("Model must be fitted before retrieving parameters.")

        param_names = MODEL_PARAMS[self.model]
        parameters = {name: param for name, param in zip(param_names, self.params_)}
        parameters["Cost"] = self.cost_
        parameters["Cost Metric"] = self.cost_function

        return parameters

    def print_error(self):
        if not hasattr(self, 'y_true') or not hasattr(self, 'y_pred'):
            raise ValueError("Model must be fitted before calculating the error.")

        absolute_error = np.abs(self.y_true - self.y_pred)
        percentage_error = (absolute_error / self.y_true) * 100
        mean_percentage_error = np.mean(percentage_error)

        print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
        print(f"Cost ({self.cost_function}): {self.cost_}")


    def plot(self, time, G_relax, savefig=False, filename="plot.png", dpi=300, file_format="png"):
        if not self.fitted_:
            raise ValueError("Model must be fitted before plotting.")
    
        import matplotlib.pyplot as plt
        # import scienceplots
        # plt.style.use(['science', 'nature', "bright"])

        # Predict G_relax using the fitted model
        G_relax_pred = self.predict(time)

        # Plot the results
        plt.figure(figsize=(3.2, 3))
        plt.plot(time, G_relax, 'o', markersize=6, label=r'$G(t)$')
        plt.plot(time, G_relax_pred, '--', color='k', lw=2, label='fit')
        plt.xscale("log")
        plt.yscale("log")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel(r'$t$ [s]', fontsize=16)
        plt.ylabel(r'$G(t)$ [Pa]', fontsize=16)
        plt.legend(fontsize=15)
        plt.grid(False)
        plt.tight_layout()

        if savefig:
            plt.savefig(filename, dpi=dpi, format=file_format, bbox_inches='tight')

        plt.show()


