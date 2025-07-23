import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
import periodic_padding_2d
import parameter
import pickle
import gc

def load_data_for_system(folder, start_frame, stop_frame, x_size, y_size):
    """
    Loads data for a given system from a specified folder and frame range.

    Args:
        folder (str): Path to the folder containing data.
        start_frame (int): Starting frame number.
        stop_frame (int): Stopping frame number.
        x_size (int): Grid size along the x-dimension.
        y_size (int): Grid size along the y-dimension.

    Returns:
        field_data0 (torch.Tensor): Tensor of potential field data.
        density_data0 (torch.Tensor): Tensor of density field data.
    """
    num_steps = stop_frame - start_frame + 1
    print("folder: ", folder)
    print("number of steps: ", num_steps)

    # Initialize PyTorch tensors in the shape [num_steps, channels, height, width]
    field_data0 = torch.zeros((num_steps, 2, x_size, y_size), dtype=torch.float32)
    density_data0 = torch.zeros((num_steps, 2, x_size, y_size), dtype=torch.float32)

    # Load only the third and fourth columns of the CSV files
    cols_to_use = [2, 3]

    # Load data from files
    field_all_data = pd.read_csv(os.path.join(folder, parameter.training_pot_name), sep=' ', header=None, usecols=cols_to_use).values
    density_all_data = pd.read_csv(os.path.join(folder, parameter.training_den_name), sep=' ', header=None, usecols=cols_to_use).values

    rows_per_frame = x_size * y_size
    for frame in range(start_frame, stop_frame + 1):
        start_idx = (frame - start_frame) * rows_per_frame
        field_data1 = field_all_data[start_idx:start_idx + rows_per_frame]
        density_data1 = density_all_data[start_idx:start_idx + rows_per_frame]
        
        # Convert data to PyTorch tensors
        new_field_data = torch.stack([
            torch.tensor(field_data1[:, 0], dtype=torch.float32).view(x_size, y_size),
            torch.tensor(field_data1[:, 1], dtype=torch.float32).view(x_size, y_size)
        ], dim=0)

        new_density_data = torch.stack([
            torch.tensor(density_data1[:, 0], dtype=torch.float32).view(x_size, y_size),
            torch.tensor(density_data1[:, 1], dtype=torch.float32).view(x_size, y_size)
        ], dim=0)
        
        # Update tensors for the current sample
        sample = frame - start_frame
        field_data0[sample] = new_field_data
        density_data0[sample] = new_density_data

    return field_data0, density_data0


def split_dataset(data, labels, test_size):
    """
    Splits data into training and testing subsets.

    Args:
        data (torch.Tensor): Input data tensor.
        labels (torch.Tensor): Corresponding labels.
        test_size (float): Fraction of data to use for testing.

    Returns:
        train_data, train_labels, test_data, test_labels (torch.Tensor): Split datasets.
    """
    total_size = data.size(0)
    test_size = int(total_size * test_size)

    shuffled_indices = torch.randperm(total_size)
    shuffled_data = data[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    test_data = shuffled_data[:test_size]
    test_labels = shuffled_labels[:test_size]
    train_data = shuffled_data[test_size:]
    train_labels = shuffled_labels[test_size:]

    return train_data, train_labels, test_data, test_labels

def standardize(data):
    """
    Standardizes data by subtracting the mean and dividing by the standard deviation.

    Args:
        data (torch.Tensor): Input data tensor.

    Returns:
        standardized_data (torch.Tensor): Standardized tensor.
        mean (torch.Tensor): Mean values used for standardization.
        std (torch.Tensor): Standard deviation values used for standardization.
    """
    mean = data.mean(dim=[0, 2, 3], keepdim=True)
    std = data.std(dim=[0, 2, 3], keepdim=True)
    return (data - mean) / std, mean, std

def load_and_preproc_data():
    """
    Loads, preprocesses, and standardizes training, validation, and test datasets.
    Saves the preprocessed datasets to files if not already saved.

    Returns:
        train_loader, val_loader (DataLoader): DataLoader objects for train, validation, and test sets.
    """
    if not os.path.exists('train_dataset.pt') or \
       not os.path.exists('val_dataset.pt') or \
       not os.path.exists('test_dataset.pt'):
        all_field_data0, all_density_data0 = None, None
        for folder in parameter.NN_training_path:
            field_data0, density_data0 = load_data_for_system(folder, parameter.start_frame, parameter.stop_frame, parameter.x_size, parameter.y_size)
            if all_field_data0 is None:
                all_field_data0 = field_data0
                all_density_data0 = density_data0
            else:
                all_field_data0 = torch.cat([all_field_data0, field_data0], dim=0)
                all_density_data0 = torch.cat([all_density_data0, density_data0], dim=0)
            del field_data0, density_data0
            gc.collect()

        # Standardize the entire dataset
        all_field_data0, mean_potential, std_potential = standardize(all_field_data0)
        all_density_data0, mean_density, std_density = standardize(all_density_data0)

        # Save normalization parameters
        normalization_params = {
            'mean_potential': mean_potential,
            'std_potential': std_potential,
            'mean_density': mean_density,
            'std_density': std_density
        }
        with open('normalization_params.pkl', 'wb') as f:
            pickle.dump(normalization_params, f)

        # Split the dataset into train, validation, and test sets
        X_train, y_train, X_test_val, y_test_val = split_dataset(all_field_data0, all_density_data0, test_size=parameter.test_val_size)
        X_val, y_val, X_test, y_test = split_dataset(X_test_val, y_test_val, test_size=parameter.val_size)
        del all_field_data0, all_density_data0, X_test_val, y_test_val
        gc.collect()

        # Apply periodic padding
        X_train_padded = periodic_padding_2d.periodic_padding_2d(X_train)
        X_val_padded = periodic_padding_2d.periodic_padding_2d(X_val)
        X_test_padded = periodic_padding_2d.periodic_padding_2d(X_test)
        y_train_padded = periodic_padding_2d.periodic_padding_2d(y_train)
        y_val_padded = periodic_padding_2d.periodic_padding_2d(y_val)
        y_test_padded = periodic_padding_2d.periodic_padding_2d(y_test)

        # Save preprocessed datasets
        torch.save((X_train_padded, y_train_padded), 'train_dataset.pt')
        torch.save((X_val_padded, y_val_padded), 'val_dataset.pt')
        torch.save((X_test_padded, y_test_padded), 'test_dataset.pt')
    else:
        # Load preprocessed datasets
        X_train_padded, y_train_padded = torch.load('train_dataset.pt')
        X_val_padded, y_val_padded = torch.load('val_dataset.pt')
        X_test_padded, y_test_padded = torch.load('test_dataset.pt')

    # Create DataLoader objects
    train_loader = DataLoader(TensorDataset(X_train_padded, y_train_padded), batch_size=32, shuffle=True, num_workers=15)
    val_loader = DataLoader(TensorDataset(X_val_padded, y_val_padded), batch_size=32, shuffle=False, num_workers=15)
    test_loader = DataLoader(TensorDataset(X_test_padded, y_test_padded), batch_size=32, shuffle=False, num_workers=15)

    return train_loader, val_loader, test_loader
