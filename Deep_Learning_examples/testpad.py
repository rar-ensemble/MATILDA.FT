import numpy as np
import tensorflow as tf
import parameter
import matplotlib.pyplot as plt

# Load and reshape data from 'pot_field_frame2.dat'
def load_and_reshape_data(filename):
    data = np.loadtxt(filename)
    x_coords = np.unique(data[:, 0])
    y_coords = np.unique(data[:, 1])
    height, width = len(y_coords), len(x_coords)
    reshaped_data = np.zeros((1, height, width, 2))
    for x, y, potA, potB in data:
        ix = np.where(x_coords == x)[0][0]
        iy = np.where(y_coords == y)[0][0]
        reshaped_data[0, iy, ix, :] = [potA, potB]
    return reshaped_data

# Numpy-based padding function
def periodic_padding_2d_np(array):
    padded_array = np.pad(array, ((0, 0), (parameter.padding, parameter.padding), (parameter.padding, parameter.padding), (0, 0)), mode='wrap')
    return padded_array

# TensorFlow-based padding function
def periodic_padding_2d_tf(array):
    padding = parameter.padding
    num_samples, height, width, num_materials = array.shape
    top_pad = tf.slice(array, [0, height-padding, 0, 0], [num_samples, padding, width, num_materials])
    bottom_pad = tf.slice(array, [0, 0, 0, 0], [num_samples, padding, width, num_materials])
    array = tf.concat([top_pad, array, bottom_pad], axis=1)
    _, height, _, _ = array.shape
    left_pad = tf.slice(array, [0, 0, width-padding, 0], [num_samples, height, padding, num_materials])
    right_pad = tf.slice(array, [0, 0, 0, 0], [num_samples, height, padding, num_materials])
    array = tf.concat([left_pad, array, right_pad], axis=2)
    return array

# Save padded data as images
def save_padded_image(padded_data, filename):
    fig, axes = plt.subplots(1, padded_data.shape[-1], figsize=(12, 4))
    for i in range(padded_data.shape[-1]):
        ax = axes[i]
        ax.imshow(padded_data[0, :, :, i], origin='lower')  # 添加origin参数
        ax.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# Main script
data = load_and_reshape_data('pot_field_frame2.dat')
np_padded = periodic_padding_2d_np(data)
tf_padded = periodic_padding_2d_tf(data).numpy()  # Convert to numpy array for consistency

save_padded_image(np_padded, 'np_padded.png')
save_padded_image(tf_padded, 'tf_padded.png')
