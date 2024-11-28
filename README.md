# pyRheo

**pyRheo** is a Python package for rheological modeling, providing tools for creep, stress relaxation, oscillation, and rotation models. This package is designed to help researchers and engineers analyze and model the behavior of viscoelastic materials.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Creep Model](#creep-model)
  - [Stress Relaxation Model](#stress-relaxation-model)
  - [Oscillation Model](#oscillation-model)
  - [Rotation Model](#rotation-model)
- [Demo Scripts](#demo-scripts)
- [GUI](#gui)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install the `pyRheo` package directly from GitHub using pip:

```sh
pip install git+https://github.com/mirandi1/pyRheo.git
```

## Usage
### Importing the package
Once the package has been installed, you can simply import its modules:

```python
from pyRheo import CreepModel, RelaxationModel, OscillationModel, RotationModel
```

### Modeling data with pyRheo
To begin modeling data, first define a model object:

```python
model = RelaxationModel(model="FractionalZenerSolidS", initial_guesses="random", 
                        num_initial_guesses=10, 
                        minimization_algorithm="Powell", 
                        mittag_leffler_type="Pade32"
                       )
```

fit the data with the model object:

```python
model.fit(time, G_relax)
```

Here, `time` and `G_relax` are Python NumPy arrays of the same size.

### Output the model results

```python
model.get_parameters()
model.print_parameters()
model.print_error()
```

### Predicting data and plotting
Plot the original data and the fitting results with 
```python
model.plot(time, G_relax)
```

or use the fitted model to predict data over a customized time range

```python
time_predict = np.logspace(np.min(np.log10(time)), np.max(np.log10(time)), 100)
G_relax_predict = model.predict(time_predict)
```

