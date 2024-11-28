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

```sh
from pyRheo import CreepModel, RelaxationModel, OscillationModel, RotationModel
```

### Modeling data with pyRheo
To begin modeling dat, first define a model object:

```sh
model = RelaxationModel(model="FractionalZenerSolidS", initial_guesses="random", 
                        num_initial_guesses=10, 
                        minimization_algorithm="Powell", 
                        mittag_leffler_type="Pade32"
                       )
```


