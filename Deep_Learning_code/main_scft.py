import torch
import numpy as np
import os
import sys
import time
import pred_density
import train_neural_network
import plot_prediction
import Gen_Init_pot
import euler_method
import pickle
from model_setup_and_training import CustomUNet

start_time = time.time()

# Set up system and device for PyTorch
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Obtain parameters from command line argument
if len(sys.argv) != 2:
    print("Usage: python main_scft.py <parameter file>")
    sys.exit(1)
parameter_file = sys.argv[1]

# Load parameters from file
parameter = {}
with open(parameter_file, 'r') as file:
    exec(file.read(), parameter)

# Load initial potential data
# Define box size and initialize the potential tensor
pot_folderpath = f"{parameter['pot_folderpath']}"
init_pot_path = os.path.join(pot_folderpath, parameter['Initial_pot'])
box_Rg_size_0 = torch.tensor(parameter['box_Rg_size'][0], dtype=torch.float32)
box_Rg_size_1 = torch.tensor(parameter['box_Rg_size'][1], dtype=torch.float32)
x_size = torch.tensor(parameter['x_size'], dtype=torch.int32)
y_size = torch.tensor(parameter['y_size'], dtype=torch.int32)

# Generate initial random potential file
output_filepath = "Random_Init.pt"
new_frame = Gen_Init_pot.generate_new_frame(box_Rg_size_0, box_Rg_size_1, x_size, y_size, std_dev=0.1)
Gen_Init_pot.save_tensor_to_file(output_filepath, new_frame)

# Load or train the neural network model
# Initialize and load model parameters
model = CustomUNet()
if not os.path.isfile("neural_network_params.pth"):
    train_neural_network.train_neural_network()

state_dict = torch.load("neural_network_params.pth")
model.load_state_dict(state_dict)

# Load normalization parameters
# Load mean and standard deviation for normalization purposes
with open('normalization_params.pkl', 'rb') as f:
    normalization_params = pickle.load(f)

mean_potential = normalization_params['mean_potential']
std_potential = normalization_params['std_potential']
mean_density = normalization_params['mean_density']
std_density = normalization_params['std_density']
current_step = -1

# Calculate intervals for the simulation box
x_interval = box_Rg_size_0 / x_size.float()
y_interval = box_Rg_size_1 / y_size.float()

################################ GPU Processing Starts Here ###########################
# Move model and necessary tensors to GPU
model.to(device)
mean_potential = normalization_params['mean_potential'].clone().detach().to(device)
std_potential = normalization_params['std_potential'].clone().detach().to(device)
mean_density = normalization_params['mean_density'].clone().detach().to(device)
std_density = normalization_params['std_density'].clone().detach().to(device)
Rho0_torch = torch.tensor(parameter['Rho0'], dtype=torch.float32, device=device)
Kappa_torch = torch.tensor(parameter['Kappa'], dtype=torch.float32, device=device)
Chi_torch = torch.tensor(parameter['Chi'], dtype=torch.float32, device=device)
N_torch = torch.tensor(parameter['N'], dtype=torch.float32, device=device)
dt_helfand_torch = torch.tensor(parameter['dt_helfand'], dtype=torch.float32, device=device)
dt_plus_torch = torch.tensor(parameter['dt_plus'], dtype=torch.float32, device=device)
dt_minus_torch = torch.tensor(parameter['dt_minus'], dtype=torch.float32, device=device)

# Create meshgrid for the box coordinates
# Define a grid of coordinates to represent the simulation space
x_coords = torch.arange(0, x_size.item(), dtype=torch.float32, device=device) * x_interval
y_coords = torch.arange(0, y_size.item(), dtype=torch.float32, device=device) * y_interval
xx, yy = torch.meshgrid(x_coords, y_coords, indexing='xy')
init_potential = new_frame.to(device)
init_density = pred_density.pred_density(init_potential, model, mean_potential, std_potential, mean_density, std_density, xx, yy, current_step)

# Reshape data for easier calculations
# Flatten the coordinate grids and potential data for processing
xx_reshaped = xx.reshape(-1)
yy_reshaped = yy.reshape(-1)
pot_Helfand_i = init_potential[:, 2].squeeze()
pot_Flory_plus_i = init_potential[:, 3].squeeze()
w_minus = init_potential[:, 4].squeeze()
Rho_A = init_density[:, 2].squeeze()
Rho_B = init_density[:, 3].squeeze()

# Euler method updates for each step
# Iterate through each time step to update potentials and densities
for step in range(parameter['steps']):
    current_step = step
    if parameter['METHOD'] == "EM":
        pot_Helfand_i, pot_Flory_plus_i, w_minus = euler_method.euler_update(
            pot_Helfand_i, pot_Flory_plus_i, w_minus, 
            Rho_A, Rho_B, Rho0_torch, Kappa_torch, Chi_torch, N_torch, 
            dt_helfand_torch, dt_plus_torch, dt_minus_torch
        )

    # Calculate potentials for components A and B
    w_plus_i = pot_Helfand_i + pot_Flory_plus_i
    pot_A = (-w_plus_i - w_minus) / N_torch
    pot_B = (-w_plus_i + w_minus) / N_torch

    # Predict updated densities using the neural network model
    updated_potential = torch.stack([xx_reshaped, yy_reshaped, pot_A, pot_B], dim=1)
    updated_density = pred_density.pred_density(updated_potential, model, mean_potential, std_potential, mean_density, std_density, xx, yy, current_step)

    # Update densities for the next iteration
    Rho_A = updated_density[:, 2]
    Rho_B = updated_density[:, 3]

# Save the results after the Euler method loop
# Convert tensors to NumPy arrays for saving
pot_Helfand_i_np = pot_Helfand_i.cpu().numpy()
pot_Flory_plus_i_np = pot_Flory_plus_i.cpu().numpy()
w_minus_np = w_minus.cpu().numpy()
xx_reshaped_np = xx_reshaped.cpu().numpy()
yy_reshaped_np = yy_reshaped.cpu().numpy()

# Stack and save final potential data
final_pot_data = np.stack([xx_reshaped_np, yy_reshaped_np, pot_Helfand_i_np, pot_Flory_plus_i_np, w_minus_np], axis=1)
final_pot_filepath = 'final_potential_pls_mns_test.dat'
np.savetxt(final_pot_filepath, final_pot_data, fmt='%f', header="x y pot_Helfand pot_Flory_plus w_minus", comments="")

# Save the updated density and potential data
updated_density_np = updated_density.cpu().numpy()
updated_potential_np = updated_potential.cpu().numpy()
density_filepath = 'final_density.txt'
potential_filepath = 'final_potential.txt'
np.savetxt(density_filepath, updated_density_np, fmt='%f', header="i j rhoA rhoB", comments="")
np.savetxt(potential_filepath, updated_potential_np, fmt='%f', header="i j potA potB", comments="")

# Visualize the data using plotting functions
plot_prediction.visualize_data(density_filepath, potential_filepath)
end_time = time.time()
print(f"Total time: {end_time - start_time} seconds")
