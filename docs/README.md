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
- [Machine Learning classifier](#machine-learning-classifier)
  - [Data standardization](#data-standardization)
  - [Multi-Layer Perceptron](#multi-layer-perceptron)
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

# Machine Learning classifier
We use a **Multi-Layer Perceptron (MLP)** to train a machine learning model that classifies a creep, stress relaxation, oscillation, or rotation dataset according to the given model labels. The training process involves preprocessing the dataset, while also splitting it into training, validation, and test sets to evaluate the model's performance objectively. Here, we summarize the preprocessing logic and the performance of the MLP models.

## Data standardization
Log10 and standard_scaler

## MLP
Multi-Layer Perceptron

# Quick start

## Installation
Via pip

## Preparing data
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



