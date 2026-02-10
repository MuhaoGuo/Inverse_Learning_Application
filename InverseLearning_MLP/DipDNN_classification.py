import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

import numpy as np
import random

# Function to set all seeds for reproducibility
def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility in convolutional layers
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

# Set the seed before initializing the model or any random operations
set_seed(42)  # You can use any seed value of your choice



# Invertible Block with triangular masks for invertibility
class Inverse_Net_maskBlock(nn.Module):
    def __init__(self, in_features):
        super(Inverse_Net_maskBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.fc2 = nn.Linear(in_features, in_features, bias=True)

        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))

    def forward(self, x):
        fc1_fwd = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        fc2_fwd = F.leaky_relu(self.fc2(fc1_fwd), negative_slope=0.01)
        return fc2_fwd

    def inverse(self, y):
        fc2_W_T = torch.linalg.inv(torch.mul(self.fc2.weight, self.mask_triu))
        fc2_inv = F.linear(F.leaky_relu(y, negative_slope=1/0.01) - self.fc2.bias, fc2_W_T)
        fc1_W_T = torch.linalg.inv(torch.mul(self.fc1.weight, self.mask_tril))
        fc1_inv = F.linear(F.leaky_relu(fc2_inv, negative_slope=1/0.01) - self.fc1.bias, fc1_W_T)
        return fc1_inv

# Define the Invertible Neural Network for feature representation
class Inverse_DNN_with_Features(nn.Module):
    def __init__(self, input_size, num_blocks, feature_dim, num_classes=10):
        super(Inverse_DNN_with_Features, self).__init__()
        self.layers = nn.ModuleList([Inverse_Net_maskBlock(input_size) for _ in range(num_blocks)])
        self.feature_fc = nn.Linear(input_size, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        features = F.leaky_relu(self.feature_fc(x))
        features = self.dropout(features)
        out = self.classifier(features)
        return out, features

    def inverse(self, features):
        x = features
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

# Load MNIST Dataset
def load_mnist(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training Function
def train(model, train_loader, test_loader, optimizer, criterion, num_epochs=10, device='cpu'):
    """
    Training loop for the invertible neural network. Tracks loss and computes accuracy.
    After each epoch, test accuracy is printed.
    """
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Flatten the inputs (28x28 image -> 784 vector if using MNIST)
            inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute predictions and training accuracy
            _, predictions = torch.max(outputs, 1)  # Get index of max log-probability
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Calculate accuracy for the epoch
        train_accuracy = 100 * correct / total
        # Test the model after each epoch and print the test accuracy
        test_accuracy = test(model, test_loader, criterion, device=device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}%")
        if scheduler is not None:
            scheduler.step(running_loss / len(train_loader))
        # # Test the model after each epoch and print the test accuracy
        # test_accuracy = test(model, test_loader, criterion, device=device)
        # print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")

# Testing Function
def test(model, test_loader, criterion, device='cpu'):
    """
    Evaluate the model on the test set. Computes accuracy and test loss.
    Returns test accuracy for reporting in the training loop.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Flatten the inputs (28x28 image -> 784 vector)
            inputs = inputs.view(inputs.size(0), -1)

            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Apply classification
            _, predictions = torch.max(outputs, 1)  # Get index of max log-probability
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = 100 * correct / total
    return test_accuracy

# Main function with device selection
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto',
                        help='Choose device: auto, cpu, cuda, or mps (for Mac M1/M2/M3)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    # Detect device: CUDA for Ubuntu GPU, MPS for MacBook with M1/M2/M3, or CPU fallback
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA (GPU) for training.")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Metal (MPS) for training on MacBook Pro with M1/M2/M3.")
        else:
            device = torch.device('cpu')
            print("Using CPU for training.")
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Load MNIST data
    train_loader, test_loader = load_mnist(batch_size=args.batch_size)


    # Load MNIST data
    train_loader, test_loader = load_mnist(batch_size=64)

    # Model configuration
    input_size = 28 * 28  # Flattened input size for MNIST (28x28 pixels)
    num_blocks = 5        # Number of invertible blocks
    feature_dim = 64      # Dimensionality of the feature space
    num_classes = 10      # 10 classes for MNIST


    # Initialize model, loss function, and optimizer
    model = Inverse_DNN_with_Features(input_size=input_size, num_blocks=num_blocks, feature_dim=feature_dim, num_classes=num_classes)
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()  # Multi-class classification loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    # Define learning rate scheduler and print its parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5,
                                                           eps=1e-6)

    # Train the model
    train(model, train_loader, test_loader, optimizer, criterion, num_epochs=args.epochs, device=device)
    # Test the model
    test(model, test_loader, criterion, device=device)
