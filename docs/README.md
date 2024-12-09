## Introduction

Welcome to pyRheo's documentation.
pyRheo is a Python package for fitting viscoelastic models to rheological data from creep, stress relaxation, oscillation, and rotation tests.

Check out the Quick Start section for more information on getting started, including an Installation guide.

> ⚠️ **Note:**  
> pyRheo is under active development. Come back for updates.

# Table of Contents

- [Introduction](#introduction)
  - [mastercurves](#mastercurves)
  - [Citing](#citing)
  - [News](#news)
- [Installation](#installation)
- [Contributing](#contributing)
- [Automated Data Superposition](#automated-data-superposition)
  - [Gaussian Process Regression](#gaussian-process-regression)
  - [Maximum A Posteriori Estimation](#maximum-a-posteriori-estimation)
- [Quick Start](#quick-start)
  - [Installation](#installation-1)
  - [Creating a master curve](#creating-a-master-curve)
  - [Adding data to a master curve](#adding-data-to-a-master-curve)
  - [Defining the coordinate transformations](#defining-the-coordinate-transformations)
  - [Superposing the data](#superposing-the-data)
  - [Plotting the results](#plotting-the-results)
- [Tutorials](#tutorials)
  - [Tutorial: Creating a Master Curve](#tutorial-creating-a-master-curve)
    - [Import Packages](#import-packages)
    - [Creating Synthetic Data](#creating-synthetic-data)
    - [Creating the Master Curve](#creating-the-master-curve)
    - [Plotting the Master Curve](#plotting-the-master-curve)
    - [Analyzing the Shift Factors](#analyzing-the-shift-factors)
  - [Tutorial: Adjusting the Gaussian Process Kernel](#tutorial-adjusting-the-gaussian-process-kernel)
    - [Changing the Gaussian Process Kernel](#changing-the-gaussian-process-kernel)
    - [Changing the Hyperparameter Bounds](#changing-the-hyperparameter-bounds)
- [Demos](#demos)
  - [Time-Temperature Superposition](#time-temperature-superposition)
    - [Import Packages](#import-packages-1)
    - [Loading the Data](#loading-the-data)
    - [Creating the Master Curve](#creating-the-master-curve-1)
    - [Plotting the Results](#plotting-the-results-1)
    - [Sensitivity to Noisy Data](#sensitivity-to-noisy-data)
      - [Adding Noise to Data](#adding-noise-to-data)
      - [Creating and Plotting the Master Curve](#creating-and-plotting-the-master-curve)
      - [Comparing Shift Factors](#comparing-shift-factors)
- [API](#api)
  - [MasterCurve](#mastercurve)
    - [MasterCurve](#mastercurve-1)
  - [Transforms](#transforms)
    - [Multiply](#multiply)
    - [PowerLawAge](#powerlawage)

