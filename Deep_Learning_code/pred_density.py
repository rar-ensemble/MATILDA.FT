import torch
import parameter
from periodic_padding_2d import periodic_padding_2d  # Ensure that this function is correctly converted to a PyTorch version
from parameter import print_flush

def pred_density(predict_data, model, mean_potential, std_potential, mean_density, std_density, xx, yy, current_step=None):
    """
    Predicts the density fields using a trained model and input potential data.

    Args:
        predict_data (torch.Tensor): Input potential data.
        model (torch.nn.Module): Trained neural network model.
        mean_potential (torch.Tensor): Mean potential for normalization.
        std_potential (torch.Tensor): Standard deviation of potential for normalization.
        mean_density (torch.Tensor): Mean density for normalization.
        std_density (torch.Tensor): Standard deviation of density for normalization.
        xx (torch.Tensor): X coordinates of the mesh grid.
        yy (torch.Tensor): Y coordinates of the mesh grid.
        current_step (int, optional): Current time step for logging purposes.

    Returns:
        torch.Tensor: Predicted density data, reshaped for output.
    """
    # Standardization functions
    def standardize(x, mean, std):
        """
        Standardizes the input tensor using the given mean and standard deviation.

        Args:
            x (torch.Tensor): Input tensor.
            mean (torch.Tensor): Mean tensor for normalization.
            std (torch.Tensor): Standard deviation tensor for normalization.

        Returns:
            torch.Tensor: Standardized tensor.
        """
        return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    def inverse_standardize(x, mean, std):
        """
        Inverse standardizes the input tensor using the given mean and standard deviation.

        Args:
            x (torch.Tensor): Standardized input tensor.
            mean (torch.Tensor): Mean tensor used during standardization.
            std (torch.Tensor): Standard deviation tensor used during standardization.

        Returns:
            torch.Tensor: Inverse standardized tensor.
        """
        return x * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    # Preprocess input potential data
    def pred_data_preparation(predict_data):
        """
        Prepares input potential data for prediction.

        Args:
            predict_data (torch.Tensor): Input potential data.

        Returns:
            torch.Tensor: Standardized input data ready for prediction.
        """
        pot_dataA = predict_data[:, 2].view(-1, 1, parameter.x_size, parameter.y_size)  # Add channel dimension
        pot_dataB = predict_data[:, 3].view(-1, 1, parameter.x_size, parameter.y_size)
        combined_data = torch.cat([pot_dataA, pot_dataB], dim=1)  # Concatenate along channel dimension
        standardized_data = standardize(combined_data, mean_potential, std_potential)
        return standardized_data

    predict_input_data = pred_data_preparation(predict_data)
    predict_input_data_padded = periodic_padding_2d(predict_input_data)

    # Perform model prediction
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        predict_output_data = model(predict_input_data_padded)

    # Crop the output to remove padding
    predict_output_data_cropped = predict_output_data[:, :, parameter.padding:-parameter.padding, parameter.padding:-parameter.padding]
    
    # Inverse standardize the output data
    pred_density_data = inverse_standardize(predict_output_data_cropped, mean_density, std_density)
    pred_density_dataA = torch.clamp(pred_density_data[:, 0, :, :], min=0)  # Clamp values to ensure non-negativity
    pred_density_dataB = torch.clamp(pred_density_data[:, 1, :, :], min=0)

    # Reshape predicted density data
    pred_density_data_reshaped = torch.stack((xx.reshape(-1), yy.reshape(-1), pred_density_dataA.reshape(-1), pred_density_dataB.reshape(-1)), dim=-1)

    # Apply scaling factor to maintain the total density
    scaling_factor = parameter.x_size * parameter.y_size * parameter.Rho0 / (torch.sum(pred_density_data_reshaped[:, 2]) + torch.sum(pred_density_data_reshaped[:, 3]))
    scaling_factors = torch.full_like(pred_density_data_reshaped[:, 2:], scaling_factor)
    scaled_columns = pred_density_data_reshaped[:, 2:] * scaling_factors

    # Concatenate the scaled density data back with coordinates
    pred_density_data_reshaped = torch.cat([pred_density_data_reshaped[:, :2], scaled_columns], dim=1)

    # Log progress
    if current_step is not None:
        if current_step == -1:
            print_flush("Initial density prediction completed.")
        elif current_step % parameter.progress_freq == 0:
            print_flush(f"Time step {current_step} prediction completed.")

    return pred_density_data_reshaped
