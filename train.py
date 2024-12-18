import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from utils import *

# Assuming the necessary functions are already defined:
# get_train_ds(), get_test_ds(), get_image()

# Custom Dataset for loading images and coordinates

import torch
import numpy as np
from torch.utils.data import Dataset

class EyeCenterDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (DataFrame): DataFrame containing the (x, y) coordinates and 'Image' column with 96x96 np.array.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image from the DataFrame (assumed to be a 96x96 uint8 array in 'Image' column)
        image = self.df.iloc[idx]['Image'].astype(np.float32) / 127.5 - 1# Get the (96, 96) NumPy array
        
        # Convert to a PyTorch tensor and add the channel dimension (1, 96, 96)
        image = np.expand_dims(image, axis=-1)  # (96, 96, 1)
        image = np.transpose(image, (2, 0, 1))  # Convert to (1, 96, 96)
        
        image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
        
        # Target coordinates (x, y) from other columns
        target = self.df.iloc[idx][['left_eye_center_x', 'left_eye_center_y']].to_numpy(dtype=np.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


# Preprocessing function
def preprocess_data(train_df, test_df):
    # Define transformations (if needed, you can add more augmentations)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1] for faster training
    ])
    
    # Create datasets
    train_dataset = EyeCenterDataset(train_df, transform=transform)
    test_dataset = EyeCenterDataset(test_df, transform=transform)
    
    # Create DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# Define the CNN model
class EyeCenterModel(nn.Module):
    def __init__(self):
        super(EyeCenterModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(256 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # Output two values (x, y)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output (x, y)
        
        return x

# Training function
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

# Save the model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, filename)

# Reload the model checkpoint
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # Initialize tqdm progress bar
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}") as tepoch:
            for images, targets in tepoch:
                images, targets = images.to(device), targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)

                # Calculate the loss
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update the progress bar description
                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (tepoch.n + 1))  # Update the loss in progress bar

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


# Main code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the datasets
train_df = get_train_ds()
test_df = get_test_ds()

# Preprocess the data and create data loaders
train_loader, test_loader = preprocess_data(train_df, test_df)

# Instantiate the model
model = EyeCenterModel().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train(model, train_loader, criterion, optimizer, device, epochs=10)
# # Train the model and save the checkpoint
# num_epochs = 50
# best_loss = float('inf')
# for epoch in range(num_epochs):
#     # Train the model for one epoch
#     train_loss = train_model(train_loader, model, criterion, optimizer, device)
#     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}')
    
#     # Evaluate the model on the test set
#     test_loss = evaluate_model(test_loader, model, criterion, device)
#     print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss}')
    
#     # Save checkpoint if the test loss improves
#     if test_loss < best_loss:
#         best_loss = test_loss
#         save_checkpoint(model, optimizer, epoch, test_loss, 'best_model.pth')

# Reload the best model
model, optimizer, epoch, loss = load_checkpoint(model, optimizer, 'best_model.pth')

# Final evaluation on the test set
final_test_loss = evaluate_model(test_loader, model, criterion, device)
print(f'Final Test Loss: {final_test_loss}')
