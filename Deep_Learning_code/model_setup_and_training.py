import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init
import parameter

# Determine whether to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomUNet(nn.Module):
    def __init__(self):
        super(CustomUNet, self).__init__()
        # Encoder part: gradually increase the number of filters to learn high-level features
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # Input channels = 2, output channels = 32
            nn.BatchNorm2d(32),  # Normalization layer to stabilize training
            nn.LeakyReLU(0.01)  # Activation function with a small negative slope
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Increase output channels to 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Increase output channels to 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25)  # Dropout to prevent overfitting
        )
        
        # Decoder part: gradually decrease the number of filters and combine features from the encoder part
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # Concatenate encoder features
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25)  # Dropout to prevent overfitting
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),  # Concatenate encoder features
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01)
        )
        self.final_conv = nn.Conv2d(32, 2, kernel_size=3, padding=1)  # Final output layer
        
        # Pooling and upsampling layers
        self.maxpool = nn.MaxPool2d(2, 2)  # Reduce feature map size by half
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample feature maps by a factor of 2
        
        self._initialize_weights()  # Initialize weights of the network
        
    def forward(self, x):
        # Encoder path: downsample input and learn feature representations
        enc1 = self.enc_conv1(x)
        pool1 = self.maxpool(enc1)
        enc2 = self.enc_conv2(pool1)
        pool2 = self.maxpool(enc2)
        enc3 = self.enc_conv3(pool2)
        
        # Decoder path: upsample and concatenate features from the encoder
        up1 = self.upsample(enc3)
        concat1 = torch.cat([up1, enc2], dim=1)  # Skip connection from encoder to retain spatial information
        dec1 = self.dec_conv1(concat1)
        
        up2 = self.upsample(dec1)
        concat2 = torch.cat([up2, enc1], dim=1)  # Another skip connection
        dec2 = self.dec_conv2(concat2)
        
        final = self.final_conv(dec2)  # Final convolution to get output
        return final.float()
    
    def _initialize_weights(self):
        # Initialize weights for convolutional and batch normalization layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
                
def model_setup_and_training(train_loader: DataLoader, val_loader: DataLoader):
    # Set up the model, optimizer, and loss function
    model = CustomUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
    criterion = nn.L1Loss()  # L1 loss function for regression tasks

    for epoch in range(parameter.epochs):
        model.train()  # Set the model to training mode
        total_loss = 0  # Initialize variable to accumulate the training loss

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

            optimizer.zero_grad()  # Clear previous gradients

            outputs = model(inputs)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Calculate loss

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            total_loss += loss.item()  # Accumulate loss

        avg_loss = total_loss / len(train_loader)  # Calculate average training loss
        print(f"Epoch {epoch+1}/{parameter.epochs}, Training Loss: {avg_loss:.4f}")  # Print training loss for the epoch
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)  # Forward pass
                val_loss += criterion(val_outputs, val_labels).item()  # Accumulate validation loss

        avg_val_loss = val_loss / len(val_loader)  # Calculate average validation loss
        print(f"Epoch {epoch+1}/{parameter.epochs}, Validation Loss: {avg_val_loss:.4f}")  # Print validation loss for the epoch

    # Save the trained model parameters to a file
    torch.save(model.state_dict(), 'neural_network_params.pth')
