# SCFT-DL_pytorch
This repository includes python codes to do SCFT with DL approach by PyTorch

The entire program is meant to use Machine Learning approach to acclerate SCFT simulation. The basic idea of this program is to replace the complex propagator computation in the classic Self-Consistent Field Theory (SCFT) Simulations with a pre-trained Neural Network and keep the physical field-theoretical update scheme of SCFT.

The functions of each code file is:
  1. main_scft.py: This is the main file of the program, coordinating the Self-Consistent Field Theory (SCFT) simulations.
     The command to start the program is: python3 main_scft.py parameters.py
     
  2. parameters.py: contains the parameters needed by the program, including physical parameters such as Flory-Huggins parameter, Helfand Potential parameter, box_size etc. and also ML parameters such as training epoch, learning rate and so on.

  3. system_setup.py: This file is responsible for setting up a GPU device.
     By default, this code allows tensorflow to take all the GPU memory during initialization.
     
  4. Gen_Init_pot.py: The name suggests it's for generating initial potentials, related to the simulation.
     For the initialization of Euler's method, it's required to have a potential field input. This code calculates the range of existing data files and generates a random potential field dataset.
     
  5. load_and_preproc_data.py: Loads training&validation data, resizing data and convert data to tf.Dataset, passing it to model.
     Also enables prefetch operation, loading training data for next batch in advance.
     
  6. model_setup_and_training.py: This file involves the architecture of the model and training details.
  
  7. euler_method.py: This file implements the Euler's method, a numerical procedure for solving ODEs(ordinary differential equations).
     Defines the euler's method update scheme with physical parameters. Updates Flory-Huggins potential and Helfand potential.

  8. pred_density.py: The purpose of this file is to predict the density field from a potential field input. It involves data-preprocessing before actual predictions and the predicting process by the pre-trained model.
     Also, it reshapes and inverse scale the predicted data, preparing the data for euler's method update.

  9. periodic_padding_2d.py: This is a utility for applying periodic padding in two dimensions for input data.

  10. plot_prediction.py: This file plots final prediction results, generating visualizations of the predicted data.
