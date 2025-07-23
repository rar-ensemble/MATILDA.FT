import random
import parameter
import torch

def sample_lines_from_file(filepath, num_samples=1000):
    """
    Randomly samples a few lines from a large file without reading the entire file 
    or calculating its total number of lines.
    """
    selected_samples = []

    # Define an approximate maximum seek position based on estimated file size
    max_seek_position = (2 + parameter.x_size * parameter.y_size) * (parameter.stop_frame - parameter.start_frame)  # Adjust this estimate as needed for your file

    with open(filepath, 'r') as f:
        for _ in range(num_samples):
            pos = random.randint(0, max_seek_position)  # Randomly select a file position
            f.seek(pos)  # Move the file pointer to the selected position
            f.readline()  # Skip to the start of the next line
            line = f.readline().strip()  # Read the line and remove leading/trailing whitespace

            if line and len(line.split()) >= 4:  # Ensure the line is non-empty and has at least 4 columns
                selected_samples.append(line)

    return selected_samples

def extract_potential_ranges_from_samples(samples):
    """
    Extracts potential ranges from sampled lines.

    Args:
        samples (list): List of sampled lines from the file.

    Returns:
        tuple: Two tuples representing the min and max of potential A and B.
    """
    all_pot_A = []
    all_pot_B = []

    for line in samples:
        data = line.split()
        if len(data) >= 4:  # Ensure there are at least four columns
            all_pot_A.append(float(data[2]))  # Extract the 3rd column (potential A)
            all_pot_B.append(float(data[3]))  # Extract the 4th column (potential B)

    # Calculate the min and max values for each potential range
    min_pot_A, max_pot_A = min(all_pot_A), max(all_pot_A)
    min_pot_B, max_pot_B = min(all_pot_B), max(all_pot_B)

    return (min_pot_A, max_pot_A), (min_pot_B, max_pot_B)

def generate_new_frame(box_Rg_size_0, box_Rg_size_1, x_size, y_size, std_dev=0.1):
    """
    Generates a new frame of data with random potential values.

    Args:
        box_Rg_size_0 (float): Box size along the first dimension.
        box_Rg_size_1 (float): Box size along the second dimension.
        x_size (int): Number of grid points along the x-dimension.
        y_size (int): Number of grid points along the y-dimension.
        std_dev (float): Standard deviation for random potential values.

    Returns:
        torch.Tensor: Generated frame as a tensor.
    """
    # Create a grid of x and y coordinates
    x = torch.linspace(0.0, box_Rg_size_0, x_size)
    y = torch.linspace(0.0, box_Rg_size_1, y_size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # Generate random potential values centered around 0 with specified std_dev
    pot_plus_Helfand = torch.randn(grid_x.shape) * std_dev
    pot_plus_Flory = torch.randn(grid_y.shape) * std_dev
    pot_minus = torch.randn(grid_x.shape) * std_dev

    # Combine generated data into a single tensor
    new_frame = torch.stack(
        (grid_x.reshape(-1), grid_y.reshape(-1), pot_plus_Helfand.reshape(-1), pot_plus_Flory.reshape(-1), pot_minus.reshape(-1)), 
        dim=-1
    )

    return new_frame

def save_tensor_to_file(filepath, tensor):
    """
    Saves a PyTorch tensor to a file.

    Args:
        filepath (str): Path to the output file.
        tensor (torch.Tensor): Tensor to save.
    """
    torch.save(tensor, filepath)

def main():
    """
    Main function to process data and generate a random initialization frame.
    """
    #####################CHANGE THE PATH HERE############################
    data_filepath = "/path/to/PotentialData.dat"  # Path to the input data file
    output_filepath = "Random_Init.pt"  # Path to save the generated PyTorch tensor

    # Sample lines from the input file
    samples = sample_lines_from_file(data_filepath)
    
    # Extract potential ranges from sampled lines
    pot_A_range, pot_B_range = extract_potential_ranges_from_samples(samples)

    # Generate new frame based on extracted potential ranges
    new_frame = generate_new_frame(
        pot_A_range[1] - pot_A_range[0],  # Box size along dimension 0
        pot_B_range[1] - pot_B_range[0],  # Box size along dimension 1
        parameter.x_size, parameter.y_size  # Grid size
    )
    
    # Save the generated frame to a file
    save_tensor_to_file(output_filepath, new_frame)

if __name__ == "__main__":
    main()
