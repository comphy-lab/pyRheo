# pyRheo

**pyRheo** is a Python package for rheological modeling, providing tools for creep, stress relaxation, oscillation, and rotation models. This package is designed to help researchers and engineers analyze and model the behavior of viscoelastic materials.

Publication of this work is coming soon. Please cite this software using the metadata in the citation file for now.

## Table of Contents
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Demos](#demos)
- [GUI](#gui)
- [Contributing](#contributing)
- [License](#license)

## Documentation
Refer to the documentation to learn more about the package and how to utilize its API.


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

## Demos
Examples of the software's functionality can be found in the demos/ directory, which showcases how to use all the modules in pyRheo.

## GUI
For a graphical user interface  (GUI) of pyRheo, follow the instructions in the gui/ directory

## Contributing
Inquiries and suggestions can be directed to isaac.mirandavaldez[at]aalto.fi or by raising an issue here.

## License
[GNU General Public Licence](https://choosealicense.com/licenses/gpl-3.0/)

## References
The data used in the demos has been collected under Creative Commons from
