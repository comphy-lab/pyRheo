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
Refer to the `docs/` folder to learn more about the package and how to utilize its API. The Documentation will be available via ReadtheDocs once the repository is public.


## Installation

Install the `pyRheo` package directly from GitHub using pip:

```sh
pip install git+https://github.com/mirandi1/pyRheo.git
```

## Usage
### Importing the package
Once the package has been installed, you can simply import its modules:

```python
from pyRheo.creep_model import CreepModel
from pyRheo.relaxation_model import RelaxationModel
from pyRheo.oscillation_model import OscillationModel
from pyRheo.rotation_model import RotationpModel
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
Examples of the software's functionality can be found in the `demos/` directory, which showcases how to use all the modules in pyRheo.

## GUI
For a graphical user interface  (GUI) of pyRheo, follow the instructions in the `gui/` directory.

## Contributing
Inquiries and suggestions can be directed to isaac.mirandavaldez[at]aalto.fi or by raising an issue here.

## License
[GNU General Public Licence](https://choosealicense.com/licenses/gpl-3.0/)

## References
The data used in the demos has been collected from:

K. Landauer, O. L. Kafka, N. H. Moser, I. Foster, B. Blaiszik and A. M. Forster, Scientific Data, 2023, 10, 356

F. A. Lavergne, P. Sollich and V. Trappe, The Journal of Chemical Physics, 2022, 156, 154901.

E. S. Epstein, L. Martinetti, R. H. Kollarigowda, O. Carey-De La Torre, J. S. Moore, R. H. Ewoldt and P. V. Braun, Journal of the American Chemical Society., 2019-02-27, 141, 3597–
3604.

I. Y. Miranda-Valdez, M. Sourroubille, T. Mäkinen, J. G. Puente-Córdova, A. Puisto, J. Koivisto and M. J. Alava, Cellulose, 2024, 31, 1545–1558.

R. G. Ricarte and S. Shanbhag, Polymer Chemistry, 2024, 15, 815–846.

K. R. Lennon, G. H. McKinley and J. W. Swan, Data-Centric Engineering, 2023, 4, e13.

R. I. Dekker, M. Dinkgreve, H. D. Cagny, D. J. Koeze, B. P. Tighe and D. Bonn, Journal of Non-Newtonian Fluid Mechanics, 2018, 261, 33–37.


Mittag-Leffler algorithms were implemented based on:

C. Zeng and Y. Q. Chen, Fractional Calculus and Applied Analysis, 2015, 18, 1492–1506.

I. O. Sarumi, K. M. Furati and A. Q. M. Khaliq, Journal of Scientific Computing, 2020, 82,
1–27.

R. Garrappa, SIAM Journal on Numerical Analysis, 2015, 53, 1350–1369.

