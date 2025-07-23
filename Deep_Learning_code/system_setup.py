import torch

def system_setup():
    """
    Sets up the system to use GPU if available, otherwise defaults to CPU.
    Prints the number of available physical GPUs and the logical GPU being used.
    """
    # Check if a GPU is available
    if torch.cuda.is_available():
        # Set PyTorch to use the first GPU
        torch.cuda.set_device(0)

        # Get the number of physical GPUs
        n_gpus = torch.cuda.device_count()
        print(f"{n_gpus} Physical GPUs, 1 Logical GPU")
    else:
        print("No GPU available, using CPU")
