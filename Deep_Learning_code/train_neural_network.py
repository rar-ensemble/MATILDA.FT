from load_and_preproc_data import load_and_preproc_data
import model_setup_and_training

def train_neural_network():
    """
    Trains the neural network by loading preprocessed data and then using the specified model setup and training function.
    """
    # Load data using the load_and_preproc_data() function
    train_loader, val_loader = load_and_preproc_data.load_and_preproc_data()

    # Train the model using the loaded training and validation data
    model_setup_and_training.model_setup_and_training(train_loader, val_loader)
