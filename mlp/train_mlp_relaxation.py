# Import necessary libraries
from relaxation_models_training import (
    MaxwellModel, SpringPot, FractionalMaxwellGel, FractionalMaxwellLiquid,
    FractionalMaxwellModel, FractionalKelvinVoigtS, FractionalKelvinVoigtD,
    FractionalKelvinVoigtModel
)
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rnd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import multiprocessing
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score  # evaluation metrics
import seaborn as sns  #data visualization library

# Load the SciencePlots style for better aesthetics, especially suited for scientific papers
import scienceplots

# Activate 'science' style from SciencePlots for clean, professional-looking plots
plt.style.use(['science', 'nature'])

def createTimeNumpyLogarithmic(start, end, n):
    startLog = math.log(start)
    endLog = math.log(end)
    array = np.array(list(range(0, n)))
    array = startLog+(endLog - startLog)/n*array
    return np.exp(array)
def createRandomError(n, std):
    return np.random.normal(loc=(1), scale=(std), size=(n,))

def DrawFigure(X, y, z=[], xlim=[], ylim=[], title = ""):
    fig = plt.figure()  #create a figure
    ax = fig.add_subplot(1, 1, 1) #add an axes object to the figure
    ax.set_title(title)
        
    if (len(xlim) != 0):
        ax.set_xlim(xlim[0], xlim[1])
    if (len(ylim) != 0):
        ax.set_ylim(ylim[0], ylim[1])
    if (len(z) != 0):
        scatter = ax.scatter(X, y, c=z, cmap='viridis', s=1)
        cbar = fig.colorbar(scatter, ax=ax, label='Z-axis (color)')
    else:
        ax.scatter(X, y, s=1)

def createData(n, error):

    models = []
    scaledData = []
    integrals = []
    scaler = StandardScaler()
    for i in range(0, n):
        #Randomly select he model:
        model = rnd.randint(0, 3)

        currentData = []
        datapointAmount = 160
        dataStart = math.pow(10, ((rnd.random()-0.5)*2*1.5-4.5))
        dataEnd = math.pow(10, ((rnd.random()-0.5)*2*5.5+2.5))
        tLog = createTimeNumpyLogarithmic(dataStart, dataEnd, datapointAmount)
        Gc = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5) # -4 to 7
        G = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5) # -4 to 7
        tau = math.pow(10, (rnd.random()- 0.5)*8) # -8 to 8
        V = math.pow(10, (rnd.random()-0.5)*2*5.5 + 1.5) #10^-4 - 10^7
        
        alpha = 0.03 + 0.94*rnd.random()
        beta = 0.03 + 0.94*rnd.random()
        if model ==4:
            beta = rnd.uniform(0.03, alpha - 0.03)
        match model:
            case 0:
                currentData = MaxwellModel(Gc, tau, tLog, error)
            case 1:
                currentData = SpringPot(V, alpha, tLog, error)
            case 2:
                currentData = FractionalMaxwellModel(tau, V, alpha, beta, tLog, error)
            case 3:
                currentData = FractionalKelvinVoigtModel(G, V, alpha, beta, tLog, error)
        
        models.append(model)
        currentData = gaussian_filter1d(currentData, sigma=4.2)
        currentDataScaled = scaler.fit_transform(currentData.reshape(-1, 1)).flatten()
        scaledData.append(currentDataScaled)
        integrals.append(integration(currentDataScaled, tLog))
    
    joblib.dump(scaler, 'scaler_relaxation.joblib')
        
    return (scaledData, models, integrals)



def integration(series, time):
    timeDiff = np.diff(time)
    result = np.sum(np.multiply(series[series.shape[0] - 1], timeDiff))
    return result

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
    tick_labels = ['M', 'SP', 'FMM', 'FKV']

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
    file_name = "confusiomatrix_relaxation.pdf"
    file_path = os.path.join(script_dir, file_name)
    fig.savefig(file_path)

tLog = createTimeNumpyLogarithmic(1e-16, 1e16, 160)
#print(tLog)
n = int(math.pow(10, 6))
num_calls = os.cpu_count()
split_value = n // num_calls


# List to store the results
scaledData = []
models = []
extraFeatures = []

if __name__ == '__main__':
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(createData, [(split_value, 0.01) for _ in range(num_calls)])
    
    for result in results:
        modelResult = result[1]
        dataResult = result[0]
        integral = result[2]
        scaledData.extend(dataResult)
        models.extend(modelResult)
        extraFeatures.extend(integral)

    # Apply PCA
    #print(scaledData)
    pca = PCA(n_components=10)  # Reduce to 10 principal components
    principal_components = pca.fit_transform(scaledData)
    # Print the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    print(f"Explained variance ratio of the PCA components: {explained_variance_ratio}")
    print(f"Cumulative explained variance: {cumulative_explained_variance}")
    joblib.dump(pca, 'pca_model_relaxation.joblib')


    #Machine learning
    y = np.array(models)
    X = principal_components 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    # Neural Network classifier
    mlp = MLPClassifier(random_state=1, max_iter=300)
    mlp.fit(X_train, y_train)


    joblib.dump(mlp, 'mlp_model_relaxation.joblib')
    y_pred = mlp.predict(X_val)
    multi_accuracy = accuracy_score(y_val, y_pred)
    multi_precision = precision_score(y_val, y_pred, average='weighted')
    print(f"Prediction accuracy: {100*multi_accuracy:.2f}%")
    print(f"Prediction precision: {100*multi_precision:.2f}%")
    title = f"Prediction accuracy: {100*multi_accuracy:.2f}%" + "\n" + f"Prediction precision: {100*multi_precision:.2f}%"
    generate_confusion_matrix(y_val, y_pred, "")




    



