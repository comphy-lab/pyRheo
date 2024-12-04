# Import necessary libraries
from oscillation_models_training import (
    MaxwellModel, SpringPot, FractionalMaxwellGel, FractionalMaxwellLiquid,
    FractionalMaxwellModel, FractionalKelvinVoigtS, FractionalKelvinVoigtD,
    FractionalKelvinVoigtModel
)
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rnd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import os
import multiprocessing

# Load the SciencePlots style for better aesthetics, especially suited for scientific papers
import scienceplots

# Activate 'science' style from SciencePlots for clean, professional-looking plots
plt.style.use(['science', 'nature'])

# Function to create an array of numbers with a logarithmic scale
def createOmegaNumpyLogarithmic(start, end, n):
    startLog = math.log(start)
    endLog = math.log(end)
    array = np.array(list(range(0, n)))
    array = startLog+(endLog - startLog)/n*array
    return np.exp(array)

# Initialize the logarithmically scaled array
tLog = createOmegaNumpyLogarithmic(0.0139, 10000, 566)

# Function to create data
def createData(n, error):
    # Initialize lists and scaler
    models = []
    scaledDataPrime = []
    scaledDataDoubleprime = []
    scaler = StandardScaler()

    # Loop to generate random data
    for i in range(0, n):
        # Generate random data points and models
        datapointAmount = 566
        dataStart = math.pow(10, ((rnd.random()-0.5)*2*1.5-4.5))
        dataEnd = math.pow(10, ((rnd.random()-0.5)*2*5.5+2.5))
        tLog = createOmegaNumpyLogarithmic(dataStart, dataEnd, datapointAmount)
        
        # Randomly select model a model
        model = rnd.randint(0, 5)  # Choices include 0-7
        currentData = []

        # Randomly generate parameters for the selected model
        Gc = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5)
        G = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5)
        tau = math.pow(10, (rnd.random()- 0.5)*2*8)
        V = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5)
        eta = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5)
        G_s = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5)
        G_p = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5)
        K = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5)
        alpha = 0.03 + 0.94*rnd.random()
        beta = 0.03 + 0.94*rnd.random()
        gama = 0.03 + 0.94*rnd.random()

        # Assign model-specific data based on the selected model
        if model ==4:
            beta = rnd.uniform(0.03, alpha - 0.03)
        match model:
            case 0:
                currentData = MaxwellModel(Gc, tau, tLog, error) # here we use tau as a parameter to generate the data easier and avoid errors
            case 1:
                currentData = SpringPot(V, alpha, tLog, error)
            case 2:
                currentData = FractionalMaxwellGel(tau, V, alpha, tLog, error)
            case 3:
                currentData = FractionalMaxwellLiquid(eta, tau, beta, tLog, error)
            case 4:
                currentData = FractionalMaxwellModel(V, tau, alpha, beta, tLog, error)
            case 5:
                currentData = FractionalKelvinVoigtModel(V, G, alpha, beta, tLog, error)
            #case 6:
            #    currentData = oscillatoryModels.KelvinVoigtModel(G, eta, tLog, error)
            #case 7:
            #    currentData = oscillatoryModels.ZenerModel(G_p, G_s, eta, tLog, error)
            #case 8:
            #    currentData = oscillatoryModels.FractionalZenerLiquid(G_p, G, eta, beta, tLog, error)
            #case 9:
            #    currentData = oscillatoryModels.FractionalZenerSolid(V, G_s, G_p, alpha, tLog, error)
            #case 10:
            #    currentData = oscillatoryModels.FractionalZenerModel(V, G, K, alpha, beta, gama, tLog, error)
                                                
        # Add the model and scaled data to respective lists
        models.append(model)
        currentData = (gaussian_filter1d(currentData[0], sigma=4.2), gaussian_filter1d(currentData[1], sigma=4.2))
        scaledDataPrime.append(scaler.fit_transform(np.log10(currentData[0]).reshape(-1, 1)).flatten())
        scaledDataDoubleprime.append(scaler.fit_transform(np.log10(currentData[1]).reshape(-1, 1)).flatten())
        #scaledDataPrime.append((np.log10(currentData[0]).reshape(-1, 1)).flatten())
        #scaledDataDoubleprime.append((np.log10(currentData[1]).reshape(-1, 1)).flatten())

    return (scaledDataPrime, scaledDataDoubleprime, models)

# Function to generate confusion matrix
def generate_confusion_matrix(y_true, y_pred, title, tick_fontsize=10, annot_fontsize=10, cbar_fontsize=10):
    """
    Generates and saves a confusion matrix.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - title: Title of the confusion matrix
    - tick_fontsize: Font size for the axis ticks
    - annot_fontsize: Font size for the annotations inside the heatmap
    - cbar_fontsize: Font size for the colorbar ticks
    """
    # Define custom tick labels
    tick_labels = ['M', 'SP', 'FMG', 'FML', 'FMM', 'FKV']

    # Create a new figure with specified size
    plt.figure(figsize=(3.5, 3.2))
    ax = plt.subplot()
    fig = plt.gcf()  # Get the current figure
    
    c_mat = confusion_matrix(y_true, y_pred)
    heatmap = sns.heatmap(c_mat, cmap="Greens", annot=True, fmt='g', ax=ax, annot_kws={"size": annot_fontsize})

    # Set axis labels and title with specified font size
    ax.set_xlabel('predicted models', fontsize=12)
    ax.set_ylabel('true models', fontsize=12)
    ax.set_title(title, fontsize=12)  # Including title with font size 12

    # Set custom tick labels on x and y axes
    ax.set_xticks([0.5 + i for i in range(len(tick_labels))])
    ax.set_yticks([0.5 + i for i in range(len(tick_labels))])
    ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
    ax.set_yticklabels(tick_labels, fontsize=tick_fontsize)

    # Customize colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)

    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "confusiomatrix_oscillation.pdf"
    file_path = os.path.join(script_dir, file_name)
    fig.savefig(file_path)

# Set the number of data points and CPU count
n = int(math.pow(10, 6))  # Number of time series
num_calls = os.cpu_count()
split_value = n // num_calls

# Main function to create and process data
if __name__ == "__main__":
    scaledDataPrime = []
    scaledDataDoublePrime = []
    models = []

    # Create a pool for multiprocessing
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(createData, [(split_value, 0.01) for _ in range(num_calls)])

    # Merge results from multiple processes
    for result in results:
        scaledDataPrime.extend(result[0])
        scaledDataDoublePrime.extend(result[1])
        models.extend(result[2])

    # Reduce dimensions using PCA
    pca = PCA(n_components=5) # G_prime needs many components compared to G_doubleprime
    principal_componentsPrime = pca.fit_transform(scaledDataPrime)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print(f"Explained variance ratio of the PCA components: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {cumulative_explained_variance}")
    joblib.dump(pca, 'pca_model_prime.joblib')

    pca = PCA(n_components=5)
    principal_componentsDoublePrime = pca.fit_transform(scaledDataDoublePrime)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print(f"Explained variance ratio of the PCA components: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {cumulative_explained_variance}")
    joblib.dump(pca, 'pca_model_double_prime.joblib')

    # Prepare data for training
    y = np.array(models)
    X = principal_componentsPrime
    X = np.hstack((X, np.array(principal_componentsDoublePrime)))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Neural Network classifier
    mlp = MLPClassifier(random_state=1, max_iter=300)
    mlp.fit(X_train, y_train)

    # Save the model
    joblib.dump(mlp, 'mlp_model_oscillation.joblib')

    # Generate predictions and evaluate the model
    y_pred = mlp.predict(X_val)
    multi_accuracy = accuracy_score(y_pred, y_val)
    multi_precision = precision_score(y_pred, y_val, average='weighted')
    print(f"Prediction accuracy: {100*multi_accuracy:.2f}%")
    print(f"Prediction precision: {100*multi_precision:.2f}%")
    title = f"Prediction accuracy: {100*multi_accuracy:.2f}%" + "\n" + f"Prediction precision: {100*multi_precision:.2f}%"
    generate_confusion_matrix(y_val, y_pred, "")
