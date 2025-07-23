# Define physical and simulation parameters


import sys
# Segment lengths for different blocks
NA1 = 4  # Length of the first A block
NB = 8   # Length of the B block
NA2 = 4  # Length of the second A block
N = NA1 + NB + NA2  # Total length of the polymer

# Fraction of A segments
f = float(NA1 + NA2) / N

# Radius of gyration (Rg) for the polymer
Rg = ((N - 1.0) / 6.0) ** 0.5

# Size of the simulation box in terms of number of Rg units
x_size = 128  # Number of grid points in the x direction
y_size = 128  # Number of grid points in the y direction
box_Rg_size = [x_size / 4, y_size / 4]  # Size of the box in terms of Rg units
box_size = [x * Rg for x in box_Rg_size]  # Physical size of the box in real units

# Define density and interaction parameters
rho0 = 1.0  # Segment density per unit volume
Chi = 2.5  # Interaction parameter
ChiN = Chi * N  # Effective interaction parameter
Kappa = 2.0  # Spring constant for Helfand term

# Calculate Rg-based density and other factors
Rho0 = rho0 * Rg ** 2  # Density factor based on Rg
Rho_factor = (box_Rg_size[0] * box_Rg_size[1]) / (x_size * y_size)  # Scaling factor for density

# Training parameters for the neural network
Training_ChiN_values = [16, 32, 48]  # ChiN values used in training

# Data preprocessing parameters
start_frame = 1  # First frame to be used for training
stop_frame = 100  # Last frame to be used for training
i_range = (1, 401)  # Range of data indices to be used
load_data_batch_size = 64  # Batch size for loading data
batch_size = 128  # Training batch size
padding = 8  # Padding used in data preprocessing
test_val_size = 0.3  # Fraction of data used for testing and validation
val_size = 0.1  # Fraction of validation data in test/val set

# Neural network training parameters
learning_rate = 0.001  # Learning rate for training
epochs = 50  # Number of epochs for training

# Prediction parameters
progress_freq = 2000  # Frequency for progress logging during prediction
num_samples = 5000  # Number of samples for predictions

# Parameters for Euler-Maruyama (EM) loop
dt_plus = 0.2  # Time step for plus variable
dt_minus = 0.1  # Time step for minus variable
dt_helfand = 0.05  # Time step for Helfand variable
steps = 20001  # Number of steps for the EM loop
METHOD = "EM"  # Method for the EM loop (Euler-Maruyama)

# Paths for training and result data
path = "/PATH/TO/TRAINING/DATA"
NN_training_path = [f"{path}ChiN{ChiN}/Triblock{i}/" for ChiN in Training_ChiN_values for i in range(i_range[0], i_range[1])]

# File names for training data
training_den_name = "DensityData.dat"  # Name of the density data file
training_pot_name = "PotentialData.dat"  # Name of the potential data file

# Paths for potential and density files
pot_folderpath = "/PATH/OF/INIT/DATAFILES"
Initial_pot = "INITIAL_POTENTIAL_DATA_NAME.dat"  # Initial potential data file
Initial_den = "INITIAL_DENSITY_DATA_NAME.dat"  # Initial density data file
predict_result_path = "/PATH/TO/SAVE/FINAL/RESULTS"
density_result_path = "FINAL_DENSITY_FILE.txt"  # File for final density results
potential_result_path = "FINAL_POTENTIAL_FILE.txt"  # File for final potential results
loss_result_path = f"LOSS_FILE_NAME_OF_{epochs}.txt"  # File for storing loss data

# Function to print and flush output for real-time feedback
def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
