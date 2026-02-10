import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_paired_images(dataset, num_images=5, figname="1over2"):
    """
    Visualizes a few samples from the paired dataset.

    Args:
        dataset (torch.utils.data.Dataset): The paired dataset.
        num_images (int): Number of images to visualize.
    """
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        original, transformed = dataset[i]
        original_np = original.squeeze().numpy()  # Convert tensor to NumPy array and remove channel dimension
        transformed_np = transformed.squeeze().numpy()

        # Display Original Image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_np, cmap='gray')
        plt.title("Original MNIST")
        plt.axis('off')

        # Display Transformed Image
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(transformed_np, cmap='gray')
        plt.title("Transformed")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"./results/{figname}.png")
    plt.show()





######### 1/2 ########################################

def generate_transformed_image(original_image, fashion_image):
    """
    Generates a new image with two components:
    - Left half: Inverted original MNIST left half (1 - x)
    - Right half: Left half of FashionMNIST image

    Args:
        original_image (torch.Tensor): The original MNIST image tensor.
        fashion_image (torch.Tensor): The FashionMNIST image tensor.

    Returns:
        torch.Tensor: The transformed image tensor of shape [1, 28, 28].
    """
    # Ensure the tensors are in the range [0, 1]
    original_image = original_image.clamp(0, 1)
    fashion_image = fashion_image.clamp(0, 1)

    # Split the MNIST image into left half (columns 0-13)
    left_half_mnist = original_image[:, :, :14]  # Shape: [1, 28, 14]
    # Split the FashionMNIST image into left half
    left_half_fashion = fashion_image[:, :, :14]  # Shape: [1, 28, 14]

    # Apply inversion: 1 - x to the MNIST left half
    left_half_mnist_inverted = 1 - left_half_mnist  # Inverted left half of MNIST

    # Concatenate inverted MNIST left half and FashionMNIST left half along the width dimension
    transformed_image = torch.cat((left_half_mnist_inverted, left_half_fashion), dim=2)  # Shape: [1, 28, 28]

    return transformed_image


class PairedMNISTFashionDataset(Dataset):
    def __init__(self, mnist_dataset, fashion_mnist_dataset, transform=None):
        """
        Initializes the PairedMNISTFashionDataset.

        Args:
            mnist_dataset (torchvision.datasets.MNIST): The original MNIST dataset.
            fashion_mnist_dataset (torchvision.datasets.FashionMNIST): The FashionMNIST dataset.
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.mnist_dataset = mnist_dataset
        self.fashion_mnist_dataset = fashion_mnist_dataset
        self.transform = transform

        # Ensure both datasets have the same length
        assert len(self.mnist_dataset) == len(self.fashion_mnist_dataset), "Datasets must be of equal length."

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Retrieve the original MNIST image and label
        mnist_image, mnist_label = self.mnist_dataset[idx]

        # Retrieve the FashionMNIST image and label
        fashion_image, fashion_label = self.fashion_mnist_dataset[idx]

        # Apply transforms if any
        if self.transform:
            mnist_image = self.transform(mnist_image)
            fashion_image = self.transform(fashion_image)

        # Generate the transformed image
        transformed_image = generate_transformed_image(mnist_image, fashion_image)

        return mnist_image, transformed_image


def get_halfMNISThalfFashion(samples = 30, batch_size = 128):

    # Define the path to store/download the data
    data_path = '../data'  # You can change this path as needed

    # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Define any additional transforms (e.g., normalization)
    additional_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the original MNIST training and testing datasets
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True)

    # Load the FashionMNIST training and testing datasets
    fashion_mnist_train = datasets.FashionMNIST(root=data_path, train=True, download=True)
    fashion_mnist_test = datasets.FashionMNIST(root=data_path, train=False, download=True)

    # Create the paired datasets
    paired_train_dataset = PairedMNISTFashionDataset(mnist_train, fashion_mnist_train, transform=additional_transforms)
    paired_test_dataset = PairedMNISTFashionDataset(mnist_test, fashion_mnist_test, transform=additional_transforms)

     #----------------------------------------------------------
    # section: Create a subset of the dataset
    from torch.utils.data import Dataset, DataLoader, Subset
    subset_indices = list(range(samples))
    paired_train_dataset = Subset(paired_train_dataset, subset_indices)
    paired_test_dataset = Subset(paired_test_dataset, subset_indices)
    # --------------------------------------------------------


    # Create DataLoaders for training and testing
    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False)


    # Visualize some paired training images
    visualize_paired_images(paired_train_dataset, num_images=5, figname = "1over2")

    return train_loader, test_loader




######### 1/2 but Vertical ########################################

def generate_transformed_image_veritcal(original_image, fashion_image):
    """
    Generates a new image with two components:
    - Left half: Inverted original MNIST left half (1 - x)
    - Right half: Left half of FashionMNIST image

    Args:
        original_image (torch.Tensor): The original MNIST image tensor.
        fashion_image (torch.Tensor): The FashionMNIST image tensor.

    Returns:
        torch.Tensor: The transformed image tensor of shape [1, 28, 28].
    """
    # Ensure the tensors are in the range [0, 1]
    original_image = original_image.clamp(0, 1)
    fashion_image = fashion_image.clamp(0, 1)

    # Split the MNIST image into left half (columns 0-13)
    left_half_mnist = original_image[:, :14, :]  # Shape: [1, 28, 14]
    # Split the FashionMNIST image into left half
    left_half_fashion = fashion_image[:, 14:, :]  # Shape: [1, 28, 14]

    # Apply inversion: 1 - x to the MNIST left half
    # left_half_mnist_inverted = 1 - left_half_mnist  # Inverted left half of MNIST
    left_half_mnist_inverted = left_half_mnist  # Inverted left half of MNIST

    # Concatenate inverted MNIST left half and FashionMNIST left half along the width dimension
    transformed_image = torch.cat((left_half_fashion, left_half_mnist_inverted), dim=1)  # Shape: [1, 28, 28]

    return transformed_image


class PairedMNISTFashionDataset_veritcal(Dataset):
    def __init__(self, mnist_dataset, fashion_mnist_dataset, transform=None):
        """
        Initializes the PairedMNISTFashionDataset.

        Args:
            mnist_dataset (torchvision.datasets.MNIST): The original MNIST dataset.
            fashion_mnist_dataset (torchvision.datasets.FashionMNIST): The FashionMNIST dataset.
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.mnist_dataset = mnist_dataset
        self.fashion_mnist_dataset = fashion_mnist_dataset
        self.transform = transform

        # Ensure both datasets have the same length
        assert len(self.mnist_dataset) == len(self.fashion_mnist_dataset), "Datasets must be of equal length."

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Retrieve the original MNIST image and label
        mnist_image, mnist_label = self.mnist_dataset[idx]

        # Retrieve the FashionMNIST image and label
        fashion_image, fashion_label = self.fashion_mnist_dataset[idx]

        # Apply transforms if any
        if self.transform:
            mnist_image = self.transform(mnist_image)
            fashion_image = self.transform(fashion_image)

        # Generate the transformed image
        transformed_image = generate_transformed_image_veritcal(mnist_image, fashion_image)

        return mnist_image, transformed_image


def get_halfMNISThalfFashion_veritcal(samples = 30, batch_size = 128):

    # Define the path to store/download the data
    data_path = '../data'  # You can change this path as needed

    # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Define any additional transforms (e.g., normalization)
    additional_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the original MNIST training and testing datasets
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True)

    # Load the FashionMNIST training and testing datasets
    fashion_mnist_train = datasets.MNIST(root=data_path, train=True, download=True)
    fashion_mnist_test = datasets.MNIST(root=data_path, train=False, download=True)

    # Create the paired datasets
    paired_train_dataset = PairedMNISTFashionDataset_veritcal(mnist_train, fashion_mnist_train, transform=additional_transforms)
    paired_test_dataset = PairedMNISTFashionDataset_veritcal(mnist_test, fashion_mnist_test, transform=additional_transforms)

     #----------------------------------------------------------
    # section: Create a subset of the dataset
    from torch.utils.data import Dataset, DataLoader, Subset
    subset_indices = list(range(samples))
    paired_train_dataset = Subset(paired_train_dataset, subset_indices)
    paired_test_dataset = Subset(paired_test_dataset, subset_indices)
    # --------------------------------------------------------


    # Create DataLoaders for training and testing
    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False)


    # Visualize some paired training images
    visualize_paired_images(paired_train_dataset, num_images=5, figname = "1over2_veritcal")

    return train_loader, test_loader



######### 1/4 ########################################

def generate_transformed_image_1over4(original_image, fashion_image):
    """
    Generates a new image with two components:
    - Bottom-left quarter: Inverted original MNIST left half (1 - x)
    - Rest: FashionMNIST image

    Args:
        original_image (torch.Tensor): The original MNIST image tensor.
        fashion_image (torch.Tensor): The FashionMNIST image tensor.

    Returns:
        torch.Tensor: The transformed image tensor of shape [1, 28, 28].
    """
    # Ensure the tensors are in the range [0, 1]
    original_image = original_image.clamp(0, 1)
    fashion_image = fashion_image.clamp(0, 1)

    # Invert the left half of the MNIST image (columns 0 to 13)
    inverted_left_mnist = 1 - original_image[:, :, :14]  # Shape: [1, 28, 14]

    # Extract the bottom half (rows 14 to 27) of the inverted left half
    inverted_bottom_left_mnist = inverted_left_mnist[:, 14:, :]  # Shape: [1, 14, 14]

    # Start with the FashionMNIST image as the base
    transformed_image = fashion_image.clone()  # Shape: [1, 28, 28]

    # Replace the bottom-left quarter with the inverted MNIST left half
    transformed_image[:, 14:, :14] = inverted_bottom_left_mnist  # Assign to bottom-left quarter

    return transformed_image



class PairedMNISTFashionDataset_1over4(Dataset):
    def __init__(self, mnist_dataset, fashion_mnist_dataset, transform=None):
        """
        Initializes the PairedMNISTFashionDataset.

        Args:
            mnist_dataset (torchvision.datasets.MNIST): The original MNIST dataset.
            fashion_mnist_dataset (torchvision.datasets.FashionMNIST): The FashionMNIST dataset.
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.mnist_dataset = mnist_dataset
        self.fashion_mnist_dataset = fashion_mnist_dataset
        self.transform = transform

        # Ensure both datasets have the same length
        assert len(self.mnist_dataset) == len(self.fashion_mnist_dataset), "Datasets must be of equal length."

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Retrieve the original MNIST image and label
        mnist_image, mnist_label = self.mnist_dataset[idx]

        # Retrieve the FashionMNIST image and label
        fashion_image, fashion_label = self.fashion_mnist_dataset[idx]

        # Apply transforms if any
        if self.transform:
            mnist_image = self.transform(mnist_image)
            fashion_image = self.transform(fashion_image)

        # Generate the transformed image
        transformed_image = generate_transformed_image_1over4(mnist_image, fashion_image)

        return mnist_image, transformed_image



def get_halfMNISThalfFashion_1over4(samples = 30, batch_size = 128):

    # Define the path to store/download the data
    data_path = '../data'  # You can change this path as needed

    # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Define any additional transforms (e.g., normalization)
    additional_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the original MNIST training and testing datasets
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True)

    # Load the FashionMNIST training and testing datasets
    fashion_mnist_train = datasets.FashionMNIST(root=data_path, train=True, download=True)
    fashion_mnist_test = datasets.FashionMNIST(root=data_path, train=False, download=True)

    # Create the paired datasets
    paired_train_dataset = PairedMNISTFashionDataset_1over4(mnist_train, fashion_mnist_train, transform=additional_transforms)
    paired_test_dataset = PairedMNISTFashionDataset_1over4(mnist_test, fashion_mnist_test, transform=additional_transforms)

     #----------------------------------------------------------
    # section: Create a subset of the dataset
    from torch.utils.data import Dataset, DataLoader, Subset
    subset_indices = list(range(samples))
    paired_train_dataset = Subset(paired_train_dataset, subset_indices)
    paired_test_dataset = Subset(paired_test_dataset, subset_indices)
    # --------------------------------------------------------


    # Create DataLoaders for training and testing
    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False)


    # Visualize some paired training images
    visualize_paired_images(paired_train_dataset, num_images=5, figname = "1over4")

    return train_loader, test_loader



######### 1/16 ########################################

def generate_transformed_image_1over16(original_image, fashion_image):
    """
    Generates a new image with two components:
    - Bottom-left quarter: Inverted original MNIST left half (1 - x)
    - Rest: FashionMNIST image

    Args:
        original_image (torch.Tensor): The original MNIST image tensor.
        fashion_image (torch.Tensor): The FashionMNIST image tensor.

    Returns:
        torch.Tensor: The transformed image tensor of shape [1, 28, 28].
    """
    # Ensure the tensors are in the range [0, 1]
    original_image = original_image.clamp(0, 1)
    fashion_image = fashion_image.clamp(0, 1)

    # Invert the left half of the MNIST image (columns 0 to 13)
    inverted_left_mnist = 1 - original_image[:, :, 11:18]  #

    # Extract the bottom half (rows 14 to 27) of the inverted left half
    inverted_bottom_left_mnist = inverted_left_mnist[:, 11:18, :]  # 

    # Start with the FashionMNIST image as the base
    transformed_image = fashion_image.clone()  # Shape: [1, 28, 28]

    # Replace the bottom-left quarter with the inverted MNIST left half
    transformed_image[:, 11:18, 11:18] = inverted_bottom_left_mnist  # Assign to bottom-left quarter

    return transformed_image



class PairedMNISTFashionDataset_1over16(Dataset):
    def __init__(self, mnist_dataset, fashion_mnist_dataset, transform=None):
        """
        Initializes the PairedMNISTFashionDataset.

        Args:
            mnist_dataset (torchvision.datasets.MNIST): The original MNIST dataset.
            fashion_mnist_dataset (torchvision.datasets.FashionMNIST): The FashionMNIST dataset.
            transform (callable, optional): Optional transforms to apply to the images.
        """
        self.mnist_dataset = mnist_dataset
        self.fashion_mnist_dataset = fashion_mnist_dataset
        self.transform = transform

        # Ensure both datasets have the same length
        assert len(self.mnist_dataset) == len(self.fashion_mnist_dataset), "Datasets must be of equal length."

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        # Retrieve the original MNIST image and label
        mnist_image, mnist_label = self.mnist_dataset[idx]

        # Retrieve the FashionMNIST image and label
        fashion_image, fashion_label = self.fashion_mnist_dataset[idx]

        # Apply transforms if any
        if self.transform:
            mnist_image = self.transform(mnist_image)
            fashion_image = self.transform(fashion_image)

        # Generate the transformed image
        transformed_image = generate_transformed_image_1over16(mnist_image, fashion_image)

        return mnist_image, transformed_image



def get_halfMNISThalfFashion_1over16(samples = 30, batch_size = 128):

    # Define the path to store/download the data
    data_path = '../data'  # You can change this path as needed

    # Ensure the results directory exists
    os.makedirs("./results", exist_ok=True)

    # Define any additional transforms (e.g., normalization)
    additional_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the original MNIST training and testing datasets
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True)

    # Load the FashionMNIST training and testing datasets
    fashion_mnist_train = datasets.FashionMNIST(root=data_path, train=True, download=True)
    fashion_mnist_test = datasets.FashionMNIST(root=data_path, train=False, download=True)

    # Create the paired datasets
    paired_train_dataset = PairedMNISTFashionDataset_1over16(mnist_train, fashion_mnist_train, transform=additional_transforms)
    paired_test_dataset = PairedMNISTFashionDataset_1over16(mnist_test, fashion_mnist_test, transform=additional_transforms)

     #----------------------------------------------------------
    # section: Create a subset of the dataset
    from torch.utils.data import Dataset, DataLoader, Subset
    subset_indices = list(range(samples))
    paired_train_dataset = Subset(paired_train_dataset, subset_indices)
    paired_test_dataset = Subset(paired_test_dataset, subset_indices)
    # --------------------------------------------------------


    # Create DataLoaders for training and testing
    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False)


    # Visualize some paired training images
    visualize_paired_images(paired_train_dataset, num_images=5, figname = "1over16")

    return train_loader, test_loader





if __name__ == "__main__":
    # get_halfMNISThalfFashion(samples = 30, batch_size = 128)
    # get_halfMNISThalfFashion_1over4(samples = 30, batch_size = 128)
    # get_halfMNISThalfFashion_1over16(samples = 30, batch_size = 128)
    get_halfMNISThalfFashion_veritcal(samples = 30, batch_size = 128)
