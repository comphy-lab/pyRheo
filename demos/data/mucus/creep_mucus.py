import time  # Import time module for measuring execution time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyRheo.creep_model import CreepModel  # For rheological modeling

# Start the timer
start_time = time.time()

# Load data
data = pd.read_csv('creep_mucus_data.csv', delimiter='\t', decimal=".")
time_data = data['Time'].values
J_creep = data['Creep Compliance'].values

# Model fitting
# Initialize the rheological model with specific fitting parameters
model = CreepModel(
    model="FractionalKelvinVoigtD",                   # Automatically selects the best model
    initial_guesses="random",       # Uses random initial guesses
    num_initial_guesses=10,         # Number of initial guesses for the optimizer
    minimization_algorithm="Powell",  # Optimization algorithm
    mittag_leffler_type="Pade63"
)

# Start the timer
start_time = time.time()

# Fit the model to the experimental data
model.fit(time_data, J_creep)


# End the timer
end_time = time.time()
execution_time = end_time - start_time

# Output model results
#print(model.predict(time_data))
model.print_parameters()
model.print_error()

print(f"Total execution time: {execution_time} seconds")

model.plot(time_data, J_creep, savefig=False)
