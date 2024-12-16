# Introduction

Welcome to pyRheo's documentation.
pyRheo is a Python package for fitting viscoelastic models to rheological data from creep, stress relaxation, oscillation, and rotation tests.

Check out the Quick Start section for more information on getting started, including an Installation guide.

> ⚠️ **Note:**  
> pyRheo is under active development. Come back for updates.

## Table of Contents

- [Introduction](#introduction)
  - [pyRheo](#pyRheo)
  - [Citing](#citing)
  - [News](#news)
  - [Installation](#installation)
  - [Contributing](#contributing)
- [Quick Start](#quick-start)
  - [Installation](#installation-1)
  - [Preparing data](#preparing-data)
  - [Fitting data](#fitting-data)
  - [Plotting results](#plotting-resutls)
  - [Analyzing results](#analyzing-results)
- [Tutorials](#tutorials)
  - [Tutorial: Fitting creep data](#tutorial-fitting-creep-data)
    - [Import Packages](#importing-packages)
    - [Loading data](#loading-data)
    - [Fitting model](#fitting-model)
    - [Plotting model](#plotting-model)
    - [Analyzing model parameters](#analyzing-model-parameters)
  - [Tutorial: Fitting stress relaxation data](#tutorial-fitting-stress-relaxation-data)
    - [Import Packages](#importing-packages)
    - [Loading data](#loading-data)
    - [Fitting model](#fitting-model)
    - [Plotting model](#plotting-model)
    - [Analyzing model parameters](#analyzing-model-parameters)
  - [Tutorial: Fitting oscillation data](#tutorial-fitting-oscillation-data)
    - [Import Packages](#importing-packages)
    - [Loading data](#loading-data)
    - [Fitting model](#fitting-model)
    - [Plotting model](#plotting-model)
    - [Analyzing model parameters](#analyzing-model-parameters)
  - [Tutorial: Fitting rotation data](#tutorial-fitting-rotation-data)
    - [Import Packages](#importing-packages)
    - [Loading data](#loading-data)
    - [Fitting model](#fitting-model)
    - [Plotting model](#plotting-model)
    - [Analyzing model parameters](#analyzing-model-parameters)
  - [Tutorial: Setting manual intial guesses and boundaries](#tutorial-setting-initial-guesses-and-boundaries)
    - [Import Packages](#importing-packages)
    - [Loading data](#loading-data)
    - [Fitting model](#fitting-model)
    - [Plotting model](#plotting-model)
    - [Analyzing model parameters](#analyzing-model-parameters)
  - [Tutorial: Setting manual intial guesses and boundaries](#tutorial-setting-initial-guesses-and-boundaries)
    - [Import Packages](#importing-packages)
    - [Loading data](#loading-data)
    - [Fitting model](#fitting-model)
- [Demos](#demos)
  - Fitting a master curve
  - Fitting a model with Mittag-Leffler function
- [Machine Learning classifier](#machine-learning-classifier)
  - [Data standardization](#data-standardization)
  - [Multi-Layer Perceptron](#multi-layer-perceptron)
- [API](#api)
  - [Creep](#creep)

## pyRheo
pyRheo is a Python package for automatically finding and fitting a viscoelastic model to describe data from creep, stress relaxation, oscillation, and rotation tests.

## Citing
This package is based on the methodology described in **pyRheo: An open-source Python package for rheology
of materials**. If you use the software and feel that it was useful for your research, plase cite this manuscript.

```
@article{miranda-valdez_niinisto_makinen_koivisto_alava_2024,
    doi = {},
    url = {},
    author = {Miranda-Valdez, Isaac Y. and Niinistö, Aaro and Mäkinen, Tero and Koivisto, Juha and Alava, Mikko J.},
    title = {pyRheo: An open-source Python package for rheology of materials},
    publisher = {arXiv},
    year = {2024},
}
```

## News
No news at the moment.

## Installation
At the moment, the way to install **pyRheo** is via cloning the GitHub repository. Later on, it will be easier with Python Package Index using pip.

## Contributing
Issues, suggestions, feedback, or any comment can be sent directly to isaac.mirandavaldez[at].aalto.fi or by rising an issues on pyRheo's GitHub.

# Quick start

## Installation
The installation will be via pip in the future. For now, the best way to use pyRheo is by cloning the respository.

## Preparing data
For using pyRheo, we recommend importing the following libraries

```sh
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
from pyRheo.creep_model import CreepModel  # For rheological modeling
```

Read csv file using pandas eg

## Fitting data
model.fit()

## Analyzing results
model.print_parameters()
model.get_parameters() they are stored as a dictionary

## Plotting results
model.plot
or model.predict and plot with matplotlib
use the modelevaluator to plot an analogous response. Eg, from the parameters found in relaxation, plot creep, etc.

# Tutorials
5 tutorials

## Fitting creep data

### Import packages

### Loading data

### Fitting model

### Plotting model

### Analyzing model parameters

# Demos

## Fitting a master curve

## Fitting a model with Mittag-Leffler function

# Machine Learning classifier
pyRheo contains pre-trained **Multi-Layer Perceptron (MLP)** models that classifies a creep, stress relaxation, oscillation, or rotation dataset in order to automatically suggest a viscoelastic model that could likely fit the dataset. Here, we summarize how the MLP models are trained.

## Data standardization
The **MLP** models are trained with synthetic data generated from randomly assigning values to parameters in the constitutive equations of several viscoelatic models. The latter results in material functions (eg., creep compliance *J(t)*) computed over a time, angular frequency, or shear rate range. The training dataset is built by generating 1 million materials functions. Over the 1 million computations, we randomly varied the models and their parameters values. In the end, we record the magnitude of the material function from every computation and its correspondig label; for example, to train the MLP classfier for creep, we record in every computation the creep compliance and the name of the model that was used to generate the creep compliance data.

Knowing that the data the user will feed to the MLP will be different in magnitude and dataset size, standarizing the training data is crucial. For the first, the result from every computation is standarized by taking its log-transformation and then by removing the mean and sclaing to unit variance. For example, 

$$
z = \log{J(t)}
$$

$$
\hat{z} = \frac{z - \mu}{s}
$$

where $z$ is the result of the log-transformation, and $\hat{z}$ is the standardized log-transformed data ($\mu$ is the mean and $s$ the standard deviation).

To deal with different dataset sizes, we use principal component analysis (PCA) for dimensionality reduction, which finds that the minimum requiried to describe a dataset is 10 components. Accordingly, the minimum size of a dataset that the user can input to the MLP is a material function with 10 records.

## MLP
The user can employ the MLP model when using the `auto` argument 

# API

## CreepModel

## Class: `CreepModel`

### Description
`CreepModel` allows you to fit and predict various rheological creep models. It supports multiple optimization methods and automatic model selection.

---

### Constructor: `__init__(self, model="Maxwell", method="RSS", initial_guesses="manual", bounds="auto", minimization_algorithm="Powell", num_initial_guesses=64, mittag_leffler_type="Pade32")`

| Parameter              | Type   | Description                    |
|------------------------|--------|--------------------------------|
| `model`                | `str`  | The rheological model to use. Default is `"Maxwell"`. Options include `"Maxwell"`, `"SpringPot"`, `"FractionalMaxwellGel"`, `"FractionalMaxwellLiquid"`, `"FractionalMaxwell"`, `"FractionalKelvinVoigtS"`, `"FractionalKelvinVoigtD"`, `"FractionalKelvinVoigt"`, `"Zener"`, `"FractionalZenerSolidS"`, `"FractionalZenerLiquidS"`, `"FractionalZenerLiquidD"`, `"FractionalZenerS"`, `"auto"` for automatic model selection. |
| `method`               | `str`  | Method for fitting the model. Default is `"RSS"`.                             |
| `initial_guesses`      | `str`  | Method for generating initial guesses. Default is `"manual"`. Other options are `"random"` and `"bayesian"`. |
| `bounds`               | `str`  | Bounds for the parameters. Default is `"auto"`.                             |
| `minimization_algorithm` | `str`  | Algorithm for minimization. Default is `"Powell"`.                          |
| `num_initial_guesses`  | `int`  | Number of initial guesses for random/bayesian methods. Default is `64`.    |
| `mittag_leffler_type`  | `str`  | Type of Mittag-Leffler function to use. Default is `"Pade32"`.              |

#### Example:
```python
creep_model = CreepModel(model="Maxwell")



