import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# 8. Define the PairedMNISTFashionMNIST Dataset
class PairedMNISTFashionMNIST(Dataset):
    def __init__(self, 
                 mnist_dataset, 
                 fashionmnist_dataset, 
                 label_mapping, 
                 fashion_label_to_indices, 
                 transform_input=None, 
                 transform_target=None, 
                 shuffle=True):
        """
        Initializes the paired dataset.

        Args:
            mnist_dataset (torch.utils.data.Dataset): The MNIST dataset subset.
            fashionmnist_dataset (torch.utils.data.Dataset): The FashionMNIST dataset subset.
            label_mapping (dict): Mapping from MNIST labels to FashionMNIST labels.
            fashion_label_to_indices (dict): Mapping from FashionMNIST labels to list of indices.
            transform_input (callable, optional): Transformations to apply to MNIST images.
            transform_target (callable, optional): Transformations to apply to FashionMNIST images.
            shuffle (bool): Whether to shuffle the FashionMNIST indices per label.
        """
        self.mnist = mnist_dataset
        self.fashionmnist = fashionmnist_dataset
        self.label_mapping = label_mapping
        self.fashion_label_to_indices = fashion_label_to_indices
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.shuffle = shuffle

        # Initialize a pointer for each label to keep track of which FashionMNIST image to use next
        self.label_pointers = {label: 0 for label in fashion_label_to_indices.keys()}

        # Optionally shuffle the FashionMNIST indices for randomness
        if self.shuffle:
            for label in self.fashion_label_to_indices:
                random.shuffle(self.fashion_label_to_indices[label])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # Get MNIST image and label
        mnist_image, mnist_label = self.mnist[idx]

        # Map MNIST label to FashionMNIST label
        fashion_label = self.label_mapping.get(mnist_label, None)
        if fashion_label is None:
            raise ValueError(f"No mapping found for MNIST label {mnist_label}")

        # Get the list of FashionMNIST indices for the mapped label
        fashion_indices = self.fashion_label_to_indices[fashion_label]

        # Get the current pointer for this label
        pointer = self.label_pointers[fashion_label]

        # If we've exhausted the list, reset the pointer and optionally reshuffle
        if pointer >= len(fashion_indices):
            pointer = 0
            if self.shuffle:
                random.shuffle(self.fashion_label_to_indices[fashion_label])
            self.label_pointers[fashion_label] = pointer

        # Get the FashionMNIST index
        fashion_idx = fashion_indices[pointer]

        # Increment the pointer
        self.label_pointers[fashion_label] += 1

        # Get the FashionMNIST image
        fashion_image, _ = self.fashionmnist[fashion_idx]  # We ignore the label here

        # Apply transformations if any
        if self.transform_input:
            mnist_image = self.transform_input(mnist_image)
        if self.transform_target:
            fashion_image = self.transform_target(fashion_image)

        return mnist_image, fashion_image


# 11. Visualization Function
def visualize_paired_samples(paired_dataset, num_samples=5):
    """
    Visualizes a few samples from the paired MNIST-FashionMNIST dataset.

    Args:
        paired_dataset (PairedMNISTFashionMNIST): The paired dataset.
        num_samples (int): Number of samples to visualize.
    """
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        mnist_image, fashion_image = paired_dataset[i]

        # Convert tensors to NumPy arrays
        mnist_np = mnist_image.squeeze().numpy()
        fashion_np = fashion_image.squeeze().numpy()

        # Plot MNIST Image
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(mnist_np, cmap='gray')
        plt.title("MNIST Input")
        plt.axis('off')

        # Plot FashionMNIST Image
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(fashion_np, cmap='gray')
        plt.title("FashionMNIST Target")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig("./results/MNIST2MNISTFashion1class.png")



def get_MNIST2MNISTFashion1class_data(samples = 30, batch_size = 128):

    # 1. Define the label mapping (only for label 0)
    mnist_to_fashionmnist_mapping = {
        0: 0   # MNIST (0) -> FashionMNIST (T-shirt/top)
    }

    # 2. Define the transformation to convert PIL Images to Tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts PIL Image to Tensor and scales pixel values to [0, 1]
    ])

    # 3. Define the path where datasets will be stored
    data_path = '../data'

    # 4. Load MNIST dataset
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    # 5. Load FashionMNIST dataset
    fashionmnist_train = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    fashionmnist_test = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)

    # 6. Function to group FashionMNIST indices by label
    def group_fashionmnist_by_label(fashionmnist_dataset):
        label_to_indices = defaultdict(list)
        for idx, (image, label) in enumerate(fashionmnist_dataset):
            label_to_indices[label].append(idx)
        return label_to_indices

    # 7. Create the label to indices mapping for FashionMNIST
    # This will be filtered to only include label 0 later
    fashion_label_to_indices_train = group_fashionmnist_by_label(fashionmnist_train)
    fashion_label_to_indices_test = group_fashionmnist_by_label(fashionmnist_test)

    # --------------------- Modification Starts Here ---------------------

    # Helper function to get indices of a specific label
    def get_label_indices(dataset, label):
        """
        Returns a list of indices for the specified label in the dataset.

        Args:
            dataset (torchvision.datasets): The dataset to search.
            label (int): The label to filter by.

        Returns:
            List[int]: List of indices where the dataset label matches the specified label.
        """
        return (dataset.targets == label).nonzero(as_tuple=True)[0].tolist()

    # Filter MNIST training and test datasets to only include label 0
    mnist_train_indices = get_label_indices(mnist_train, 0)
    mnist_test_indices = get_label_indices(mnist_test, 0)

    mnist_train_subset = Subset(mnist_train, mnist_train_indices)
    mnist_test_subset = Subset(mnist_test, mnist_test_indices)

    # Filter FashionMNIST training and test datasets to only include label 0
    fashionmnist_train_indices = get_label_indices(fashionmnist_train, 0)
    fashionmnist_test_indices = get_label_indices(fashionmnist_test, 0)

    fashionmnist_train_subset = Subset(fashionmnist_train, fashionmnist_train_indices)
    fashionmnist_test_subset = Subset(fashionmnist_test, fashionmnist_test_indices)

    # Update the label to indices mapping for FashionMNIST subsets
    fashion_label_to_indices_train = group_fashionmnist_by_label(fashionmnist_train_subset)
    fashion_label_to_indices_test = group_fashionmnist_by_label(fashionmnist_test_subset)

    # Since we're only dealing with label 0, ensure the mapping exists
    if 0 not in fashion_label_to_indices_train:
        raise ValueError("No FashionMNIST samples with label 0 found in the training set.")
    if 0 not in fashion_label_to_indices_test:
        raise ValueError("No FashionMNIST samples with label 0 found in the test set.")

    # --------------------- Modification Ends Here ---------------------

    # 9. Create the paired datasets using the filtered subsets
    paired_train_dataset = PairedMNISTFashionMNIST(
        mnist_dataset=mnist_train_subset,  # Use the filtered MNIST training subset
        fashionmnist_dataset=fashionmnist_train_subset,  # Use the filtered FashionMNIST training subset
        label_mapping=mnist_to_fashionmnist_mapping,  # Only includes mapping for label 0
        fashion_label_to_indices=fashion_label_to_indices_train,
        transform_input=None,  # Already transformed with ToTensor()
        transform_target=None, # Already transformed with ToTensor()
        shuffle=True
    )

    paired_test_dataset = PairedMNISTFashionMNIST(
        mnist_dataset=mnist_test_subset,  # Use the filtered MNIST test subset
        fashionmnist_dataset=fashionmnist_test_subset,  # Use the filtered FashionMNIST test subset
        label_mapping=mnist_to_fashionmnist_mapping,  # Only includes mapping for label 0
        fashion_label_to_indices=fashion_label_to_indices_test,
        transform_input=None,
        transform_target=None,
        shuffle=True
    )


    #----------------------------------------------------------
    # section: Create a subset of the dataset
    subset_indices = list(range(samples))
    paired_train_dataset = Subset(paired_train_dataset, subset_indices)
    paired_test_dataset = Subset(paired_test_dataset, subset_indices)
    # --------------------------------------------------------

    # 10. Create DataLoaders for the paired datasets
    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False)


    # 12. Visualize some paired training samples
    # visualize_paired_samples(paired_train_dataset, num_samples=5)

    return train_loader, test_loader




def get_MNIST2MNISTFashion1sample_data(samples  = 10, batch_size = 128):
    # 1. Define the label mapping
    mnist_to_fashionmnist_mapping = {
        0: 0,  # MNIST (0) -> FashionMNIST (T-shirt/top)
        1: 1,  # MNIST (1) -> FashionMNIST (Trouser)
        2: 2,  # MNIST (2) -> FashionMNIST (Pullover)
        3: 3,  # MNIST (3) -> FashionMNIST (Dress)
        4: 4,  # MNIST (4) -> FashionMNIST (Coat)
        5: 5,  # MNIST (5) -> FashionMNIST (Sandal)
        6: 6,  # MNIST (6) -> FashionMNIST (Shirt)
        7: 7,  # MNIST (7) -> FashionMNIST (Sneaker)
        8: 8,  # MNIST (8) -> FashionMNIST (Bag)
        9: 9   # MNIST (9) -> FashionMNIST (Ankle boot)
    }

    # 2. Define the transformation to convert PIL Images to Tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts PIL Image to Tensor and scales pixel values to [0, 1]
    ])

    # 3. Define the path where datasets will be stored
    data_path = '../data'

    # 4. Load MNIST dataset
    mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    # 5. Load FashionMNIST dataset
    fashionmnist_train = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    fashionmnist_test = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)

    # 6. Function to group FashionMNIST indices by label
    def group_fashionmnist_by_label(fashionmnist_dataset):
        label_to_indices = defaultdict(list)
        for idx, (image, label) in enumerate(fashionmnist_dataset):
            label_to_indices[label].append(idx)
        return label_to_indices

    # 7. Create the label to indices mapping for FashionMNIST
    fashion_label_to_indices_train = group_fashionmnist_by_label(fashionmnist_train)
    fashion_label_to_indices_test = group_fashionmnist_by_label(fashionmnist_test)

    # 8. Select one MNIST index per label for training
    selected_mnist_train_indices = []
    for label in range(10):
        # Find the first index with the current label
        for idx, (img, lbl) in enumerate(mnist_train):
            if lbl == label:
                selected_mnist_train_indices.append(idx)
                break
        else:
            raise ValueError(f"No MNIST samples found for label {label}")

    # Create a Subset of MNIST training data with selected indices
    subset_mnist_train = Subset(mnist_train, selected_mnist_train_indices)

    # 9. Create the paired training dataset with only one sample per label
    paired_train_dataset = PairedMNISTFashionMNIST(
        mnist_dataset=subset_mnist_train,
        fashionmnist_dataset=fashionmnist_train,
        label_mapping=mnist_to_fashionmnist_mapping,
        fashion_label_to_indices=fashion_label_to_indices_train,
        transform_input=None,  # Already transformed with ToTensor()
        transform_target=None, # Already transformed with ToTensor()
        shuffle=True
    )

    # Repeat the same process for the test dataset if needed
    selected_mnist_test_indices = []
    for label in range(10):
        for idx, (img, lbl) in enumerate(mnist_test):
            if lbl == label:
                selected_mnist_test_indices.append(idx)
                break
        else:
            raise ValueError(f"No MNIST samples found for label {label} in test set")

    subset_mnist_test = Subset(mnist_test, selected_mnist_test_indices)

    paired_test_dataset = PairedMNISTFashionMNIST(
        mnist_dataset=subset_mnist_test,
        fashionmnist_dataset=fashionmnist_test,
        label_mapping=mnist_to_fashionmnist_mapping,
        fashion_label_to_indices=fashion_label_to_indices_test,
        transform_input=None,
        transform_target=None,
        shuffle=True
    )

    # 10. Create DataLoaders for the paired datasets
    train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(paired_test_dataset, batch_size=batch_size, shuffle=False)

    # 11. Visualize the paired training samples
    visualize_paired_samples(paired_train_dataset, num_samples=10)  # Set to 10

    return train_loader, test_loader



if __name__ == "__main__":

    # get_MNIST2MNISTFashion1class_data(samples = 30, batch_size = 128)
    get_MNIST2MNISTFashion1sample_data(samples = 10, batch_size = 128)


