import torch
import parameter  # Make sure the parameter module is correctly imported

def periodic_padding_2d(array):
    """
    Applies periodic padding to a 4D tensor representing a batch of 2D arrays.
    The padding is applied such that the edges wrap around, simulating periodic boundary conditions.

    Args:
        array (torch.Tensor): A 4D tensor with shape [batch_size, channels, height, width].

    Returns:
        torch.Tensor: A 4D tensor with periodic padding applied.
    """
    padding = parameter.padding  # Retrieve padding value from the parameter file

    # Get dimensions of the original array: [batch_size, channels, height, width]
    _, _, height, width = array.shape

    # Apply padding at the top and bottom: copy the top and bottom rows
    top_pad = array[:, :, -padding:, :]  # Take the last 'padding' rows
    bottom_pad = array[:, :, :padding, :]  # Take the first 'padding' rows
    vertical_padded = torch.cat([top_pad, array, bottom_pad], dim=2)

    # Apply padding at the left and right: copy the leftmost and rightmost columns
    left_pad = vertical_padded[:, :, :, -padding:]  # Take the last 'padding' columns
    right_pad = vertical_padded[:, :, :, :padding]  # Take the first 'padding' columns
    array_padded = torch.cat([left_pad, vertical_padded, right_pad], dim=3)

    return array_padded
