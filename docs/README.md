## Introduction

Welcome to pyRheo's documentation.
pyRheo is a Python package for fitting viscoelastic models to rheological data from creep, stress relaxation, oscillation, and rotation tests.

Check out the Quick Start section for more information on getting started, including an Installation guide.

> ⚠️ **Note:**  
> pyRheo is under active development. Come back for updates.

# Table of Contents

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
  - [Analyzing results](#analyzing-results)
  - [Plotting results](#plotting-resutls)
- [Tutorials](#tutorials)
  - [Tutorial: Fitting creep data](#tutorial-fitting-creep-data)
    - [Import Packages](#import-packages)
    - [Creating Synthetic Data](#creating-synthetic-data)
    - [Creating the Master Curve](#creating-the-master-curve)
    - [Plotting the Master Curve](#plotting-the-master-curve)
    - [Analyzing the Shift Factors](#analyzing-the-shift-factors)
  - [Tutorial: Fitting stress relaxation data](#tutorial-fitting-stress-relaxation-data)
    - [Import Packages](#import-packages)
    - [Creating Synthetic Data](#creating-synthetic-data)
    - [Creating the Master Curve](#creating-the-master-curve)
    - [Plotting the Master Curve](#plotting-the-master-curve)
    - [Analyzing the Shift Factors](#analyzing-the-shift-factors)
  - [Tutorial: Fitting oscillation data](#tutorial-fitting-oscillation-data)
    - [Import Packages](#import-packages)
    - [Creating Synthetic Data](#creating-synthetic-data)
    - [Creating the Master Curve](#creating-the-master-curve)
    - [Plotting the Master Curve](#plotting-the-master-curve)
    - [Analyzing the Shift Factors](#analyzing-the-shift-factors)
  - [Tutorial: Fitting rotation data](#tutorial-fitting-rotation-data)
    - [Import Packages](#import-packages)
    - [Creating Synthetic Data](#creating-synthetic-data)
    - [Creating the Master Curve](#creating-the-master-curve)
    - [Plotting the Master Curve](#plotting-the-master-curve)
    - [Analyzing the Shift Factors](#analyzing-the-shift-factors)
  - [Tutorial: Setting manual intial guesses and boundaries](#tutorial-setting-initial-guesses-and-boundaries)
    - [Import Packages](#import-packages)
    - [Creating Synthetic Data](#creating-synthetic-data)
    - [Creating the Master Curve](#creating-the-master-curve)
    - [Plotting the Master Curve](#plotting-the-master-curve)
    - [Analyzing the Shift Factors](#analyzing-the-shift-factors)
- [Demos](#demos)

- [API](#api)
  - [pyRheo](#pyRheo)
    - [pyRheo](#pyRheo-1)

