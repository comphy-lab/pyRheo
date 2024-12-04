# Import necessary libraries
from creep_models_training import (
    MaxwellModel, SpringPot, FractionalMaxwellGel, FractionalMaxwellLiquid,
    FractionalMaxwellModel, FractionalKelvinVoigtS, FractionalKelvinVoigtD,
    FractionalKelvinVoigtModel
)
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rnd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import multiprocessing
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import seaborn as sns
from scipy.signal import lfilter
from scipy.signal import savgol_filter
import os
import scienceplots

# Set the style for plots
plt.style.use(['science', 'nature'])

# Function to draw scatter plots with optional color map
def DrawFigure(X, y, z=[], xlim=[], ylim=[], title=""):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    
    if len(xlim) != 0:
        ax.set_xlim(xlim[0], xlim[1])
    if len(ylim) != 0:
        ax.set_ylim(ylim[0], ylim[1])
    if len(z) != 0:
        scatter = ax.scatter(X, y, c=z, cmap='viridis', s=1)
        cbar = fig.colorbar(scatter, ax=ax, label='Z-axis (color)')
    else:
        ax.scatter(X, y, s=1)

# Function to create logarithmically spaced time array
def createTimeNumpyLogarithmic(start, end, n):
    startLog = math.log(start)
    endLog = math.log(end)
    array = np.array(range(0, n))
    array = startLog + (endLog - startLog) / n * array
    return np.exp(array)

# Function to calculate derivative of y with respect to t
def calculateDerivative(t, y):
    return np.gradient(y, t)

# Function to extract constant term from linear fit of first 3 data points
def extractConstantTerm(t, y):
    coefficients = np.polyfit(t, y, 1)
    poly_func = np.poly1d(coefficients)
    return poly_func([0])[0]

# Function to create dataset for different creep models
def createData(n, error):
    models = []
    derivatives = []
    scaledData = []
    scaler = StandardScaler()
    derivativeStds = []
    constantTerms = []
    for i in range(n):
        try:
            # Randomly select model and parameters
            model = rnd.randint(0, 3)
            datapointAmount = 160
            dataStart = math.pow(10, ((rnd.random() - 0.5) * 2 * 1.5 - 4.5))
            dataEnd = math.pow(10, ((rnd.random() - 0.5) * 2 * 5.5 + 2.5))
            tLog = createTimeNumpyLogarithmic(dataStart, dataEnd, datapointAmount)

            G = math.pow(10, (rnd.random() - 0.5) * 2 * 5.5 + 1.5)
            tau = math.pow(10, (rnd.random() - 0.5) * 16)
            V = math.pow(10, (rnd.random() - 0.5) * 2 * 5.5 + 1.5)
            eta = math.pow(10, (rnd.random() - 0.5) * 2 * 5.5 + 1.5)
            alpha = 0.03 + 0.94 * rnd.random()
            beta = 0.03 + 0.94 * rnd.random()
            if model in [4, 5]:
                beta = rnd.uniform(0.03, alpha - 0.03)
            
            # Generate data based on the selected model
            match model:
                case 0:
                    currentData = MaxwellModel(G, eta, tLog, error)
                case 1:
                    currentData = SpringPot(V, alpha, tLog, error)
                #case 2:
                #    currentData = FractionalMaxwellGel(V, G, alpha, tLog, error)
                #case 3:
                #    currentData = FractionalMaxwellLiquid(G, eta, beta, tLog, error)
                case 2:
                    currentData = FractionalMaxwellModel(G, V, alpha, beta, tLog, error)
                #case 5:
                #    currentData = FractionalKelvinVoigtS(V, tau, alpha, tLog, error)
                #case 6:
                #    currentData = FractionalKelvinVoigtD(tau, eta, beta, tLog, error)
                case 3:
                    currentData = FractionalKelvinVoigtModel(tau, V, alpha, beta, tLog, error)
            
            models.append(model)
            if error != 0:
                currentData = gaussian_filter1d(currentData, sigma=4.2)

            scaledCurrentDataNoLog = scaler.fit_transform(currentData.reshape(-1, 1)).flatten()
            currentDataNoLog = currentData
            currentData = np.log10(currentData)
            derivative = calculateDerivative(tLog, scaledCurrentDataNoLog)
            scaledData.append(currentData)
            derivatives.append(np.mean(derivative))
            derivativeStds.append(np.std(derivative))
            constantTerms.append(extractConstantTerm(tLog[:3], currentDataNoLog[:3]))
        except OverflowError:
            pass
    return (scaledData, models, derivatives, derivativeStds, constantTerms)

# Function to generate and save confusion matrix
def generate_confusion_matrix(y_true, y_pred, title, tick_fontsize=10, annot_fontsize=10, cbar_fontsize=10):
    tick_labels = ['M', 'SP', 'FMM', 'FKV']
    plt.figure(figsize=(3.5, 3.2))
    ax = plt.subplot()
    fig = plt.gcf()
    c_mat = confusion_matrix(y_true, y_pred)
    heatmap = sns.heatmap(c_mat, cmap="Greens", annot=True, fmt='g', ax=ax, annot_kws={"size": annot_fontsize})
    ax.set_xlabel('predicted models', fontsize=12)
    ax.set_ylabel('true models', fontsize=12)
    #ax.set_title(title, fontsize=12)
    ax.set_xticks([0.5 + i for i in range(len(tick_labels))])
    ax.set_yticks([0.5 + i for i in range(len(tick_labels))])
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "confusiomatrix_creep.pdf")
    fig.savefig(file_path)

# Main execution block
if __name__ == '__main__':
    n = int(math.pow(10, 6))
    num_calls = os.cpu_count()
    split_value = n // num_calls
    data, models, derivatives, derivativeStds, constantTerms = [], [], [], [], []

    # Parallel data generation
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(createData, [(split_value, 0.01) for _ in range(num_calls)])
    
    for result in results:
        data.extend(result[0])
        models.extend(result[1])
        derivatives.extend(result[2])
        derivativeStds.extend(result[3])
        constantTerms.extend(result[4])

    # PCA for dimensionality reduction
    pca = PCA(n_components=10)
    principal_components = pca.fit_transform(data)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    joblib.dump(pca, 'pca_model_creep.joblib')

    # Model training and evaluation
    y = np.array(models)
    X = principal_components
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    mlp = MLPClassifier(random_state=1, max_iter=300)
    mlp.fit(X_train, y_train)
    joblib.dump(mlp, 'mlp_model_creep.joblib')

    # Predictions and metrics
    y_pred = mlp.predict(X_val)
    multi_accuracy = accuracy_score(y_val, y_pred)
    multi_precision = precision_score(y_val, y_pred, average='weighted')
    print(f"Prediction accuracy: {multi_accuracy}")
    print(f"Prediction precision (weighted): {multi_precision}")
    
    # Generate and save confusion matrix
    generate_confusion_matrix(y_val, y_pred, title='Confusion Matrix: Validation')

