import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


def generate_transformed_image(original_image):
    """
    Generates a new image with two components:
    - Left half: Inverted original left half (1 - x)
    - Right half: Original left half

    Args:
        original_image (PIL Image or torch Tensor): The original MNIST image.

    Returns:
        torch.Tensor: The transformed image tensor of shape [1, 28, 28].
    """
    # Convert PIL Image to Tensor if necessary
    if isinstance(original_image, Image.Image):
        original_image = transforms.ToTensor()(original_image)  # Shape: [1, 28, 28]
    elif isinstance(original_image, np.ndarray):
        original_image = torch.from_numpy(original_image).float() / 255.0
    elif isinstance(original_image, torch.Tensor):
        # Ensure the tensor is in the range [0, 1]
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
    else:
        raise TypeError("Unsupported image type. Must be PIL Image, NumPy array, or torch.Tensor.")
    
    # Split the image into left half (columns 0-13)
    left_half = original_image[:, :, :14]  # Shape: [1, 28, 14]
    
    # Apply inversion: 1 - x
    left_half_inverted = 1 - left_half  # Inverted left half
    
    # Concatenate inverted left half and original left half along the width dimension
    transformed_image = torch.cat((left_half_inverted, left_half), dim=2)  # Shape: [1, 28, 28]
    
    return transformed_image


def visualize_paired_images(dataset, num_images=5):
    """
    Visualizes a few samples from the paired dataset.

    Args:
        dataset (torch.utils.data.Dataset): The paired dataset.
        num_images (int): Number of images to visualize.
    """
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        original, transformed = dataset[i]
        print(original)
        print("asasaassa")
        print(transformed)

        original_np = original.squeeze().numpy()  # Convert tensor to NumPy array and remove channel dimension
        transformed_np = transformed.squeeze().numpy()
        
        # Display Original Image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_np, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Display Transformed Image
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(transformed_np, cmap='gray')
        plt.title("Transformed")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("./results/rawdata.png")

class PairedMNISTDataset(Dataset):
    def __init__(self, original_dataset, transform=None, target_transform=None):
        """
        Initializes the PairedMNISTDataset.

        Args:
            original_dataset (torchvision.datasets.MNIST): The original MNIST dataset.
            transform (callable, optional): Optional transforms to apply to the original image.
            target_transform (callable, optional): Optional transforms to apply to the transformed image.
        """
        self.original_dataset = original_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Retrieve the original image and label
        original_image, label = self.original_dataset[idx]
        
        # Generate the transformed image
        transformed_image = generate_transformed_image(original_image)
        
        # Apply transforms if any
        if self.transform:
            original_image = self.transform(original_image)
        if self.target_transform:
            transformed_image = self.target_transform(transformed_image)
        
        return original_image, transformed_image



if __name__ =="__main__":
    # Define the path to store/download the MNIST data
    data_path = '../data'  # You can change this path as needed

    # Define any additional transforms (e.g., normalization)
    additional_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the original MNIST training and testing datasets
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True)

    # Create the paired datasets
    paired_mnist_train = PairedMNISTDataset(mnist_train, transform=additional_transforms)
    paired_mnist_test = PairedMNISTDataset(mnist_test, transform=additional_transforms)

    # Define batch size
    batch_size = 64

    # Create DataLoaders for training and testing
    train_loader = DataLoader(paired_mnist_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(paired_mnist_test, batch_size=batch_size, shuffle=False)

    # Visualize some paired training images
    visualize_paired_images(paired_mnist_train, num_images=5)
