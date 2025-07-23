import numpy as np
import matplotlib.pyplot as plt
import parameter
import os

def visualize_data(density_filepath, potential_filepath, step=None):
    """
    Visualizes density and potential data from given file paths.

    Args:
        density_filepath (str): Path to the density data file.
        potential_filepath (str): Path to the potential data file.
        step (int, optional): Step number to include in the titles of the plots.
    """
    def load_data_from_file(filepath):
        """
        Loads data from a given file and reshapes it according to the x and y sizes from the parameter file.

        Args:
            filepath (str): Path to the file to be loaded.

        Returns:
            tuple: Reshaped x, y, and field arrays.
        """
        data = np.loadtxt(filepath, skiprows=1)  # Load data while skipping the header row
        x_size = parameter.x_size  # Get the x dimension size from the parameter file
        y_size = parameter.y_size  # Get the y dimension size from the parameter file
        x = data[:, 0].reshape(x_size, y_size)  # Reshape the x data
        y = data[:, 1].reshape(x_size, y_size)  # Reshape the y data
        field = data[:, 2].reshape(x_size, y_size)  # Reshape the field data (density or potential)
        return x, y, field

    # Load density and potential data from the specified file paths
    x_density, y_density, density_data_A = load_data_from_file(density_filepath)
    x_potential, y_potential, potential_data_A = load_data_from_file(potential_filepath)

    # Visualize Density_A
    plt.imshow(np.flipud(density_data_A), origin='lower', extent=(x_density.min(), x_density.max(), y_density.min(), y_density.max()), aspect='auto', vmin=0.0, vmax=2.6)
    title = 'Density_A' + (f' at step {step}' if step is not None else '')
    plt.title(title, fontsize=20)
    plt.colorbar(label='Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    filename = f"Density_A_step_{step}.png" if step is not None else "Density_A.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the plot to a file
    plt.show()  # Display the plot

    # Visualize Potential_A
    plt.imshow(np.flipud(potential_data_A), origin='lower', extent=(x_potential.min(), x_potential.max(), y_potential.min(), y_potential.max()), aspect='auto')
    title = 'Potential_A' + (f' at step {step}' if step is not None else '')
    plt.title(title, fontsize=20)
    plt.colorbar(label='Potential')
    plt.xlabel('X')
    plt.ylabel('Y')
    filename = f"Potential_A_step_{step}.png" if step is not None else "Potential_A.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the plot to a file
    plt.show()  # Display the plot

# Example call to the function (uncomment to use)
# visualize_data("density_filepath.txt", "potential_filepath.txt", step=1)
