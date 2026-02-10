import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL.ImageOps
import numpy as np

# Function to invert colors
def invert_colors(image):
    """Invert colors of an image (color inversion)"""
    # Ensure the image is in PIL format
    if not isinstance(image, PIL.Image.Image):
        image = transforms.ToPILImage()(image)
    inverted_image = PIL.ImageOps.invert(image)
    return inverted_image

# Generate dataset with transformation applied
def generate_dataset(transform_type='invert_colors', data=None):
    """Generate datasets with chosen transformation applied"""
    transform_functions = {
        'invert_colors': invert_colors,
    }
    transform_fn = transform_functions.get(transform_type)

    if transform_fn is None:
        raise ValueError(f"Invalid transform type: {transform_type}")

    # Apply transformation to the entire dataset
    transformed_dataset = []
    for idx in range(len(data)):
        image, label = data[idx]
        transformed_image = transform_fn(image)
        transformed_dataset.append((image, transformed_image))
    return transformed_dataset

# Prepare the dataset for neural network input-output pairs
def prepare_for_nn(dataset):
    """Prepare the dataset in the format required for NN (inputs and outputs as tensors)"""
    inputs = []
    outputs = []
    for original_image, transformed_image in dataset:
        input_tensor = transforms.ToTensor()(original_image)  # Convert to tensor [C, H, W], values in [0,1]
        output_tensor = transforms.ToTensor()(transformed_image)
        inputs.append(input_tensor)
        outputs.append(output_tensor)

    inputs = torch.stack(inputs)  # Stack into batch
    outputs = torch.stack(outputs)
    return inputs, outputs

# Visualize some examples from the generated dataset
def visualize_transformed_examples(dataset, num_examples=5):
    plt.figure(figsize=(10, 4))
    for i in range(num_examples):
        original_image, transformed_image = dataset[i]
        plt.subplot(2, num_examples, i+1)
        plt.imshow(original_image)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_examples, i+1+num_examples)
        plt.imshow(transformed_image)
        plt.title('Transformed')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("./results/cifar10_raw_data.png")

def load_cifar10_convert_data(batch_size=128, samples=30):
    # Load CIFAR-10 dataset
    data_path = '../data/'
    # No transformation needed here; we'll handle it in generate_dataset
    cifar10_data_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
    cifar10_data_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)

    # Create a subset of the dataset
    subset_indices = list(range(samples))
    cifar10_data_train = Subset(cifar10_data_train, subset_indices)
    cifar10_data_test = Subset(cifar10_data_test, subset_indices)

    # Generate dataset with color inversion
    transformed_dataset_train = generate_dataset(transform_type='invert_colors', data=cifar10_data_train)
    transformed_dataset_test = generate_dataset(transform_type='invert_colors', data=cifar10_data_test)

    # Visualize the transformed examples
    visualize_transformed_examples(transformed_dataset_train)
    # visualize_transformed_examples(transformed_dataset_test)  # Uncomment to visualize test examples

    # Prepare the dataset for training the neural network
    inputs_train, outputs_train = prepare_for_nn(transformed_dataset_train)
    inputs_test, outputs_test = prepare_for_nn(transformed_dataset_test)

    # Prepare DataLoader
    train_tensor = TensorDataset(inputs_train, outputs_train)
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False)

    test_tensor = TensorDataset(inputs_test, outputs_test)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def visualize_results_color(model, test_loader, device, num_samples=3, save_path="./res.png"):
    model.eval()
    image_l = 32 
    samples_visualized = 0
    plt.figure(figsize=(12, num_samples * 3))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass: input -> inverted
            outputs = model(inputs)

            # Inverse pass: inverted -> input
            reconstructed_inputs = model.inverse(outputs)

            # Inverse pass: input -> some output
            inverse_of_y = model.inverse(targets)

            for i in range(inputs.size(0)):
                if samples_visualized >= num_samples:
                    break
                
                # Original Input x
                original = inputs[i].cpu().numpy().reshape(3, image_l, image_l)

                # target y 
                target = targets[i].cpu().numpy().reshape(3, image_l, image_l)

                # Forward Output  f(x)
                forward_output = outputs[i].cpu().numpy().reshape(3, image_l, image_l)

                # Inverse of f(x)
                inverse_forward = reconstructed_inputs[i].cpu().numpy().reshape(3, image_l, image_l)

                # Inverse of y
                inverse_input = inverse_of_y[i].cpu().numpy().reshape(3, image_l, image_l)

                # Plotting
                titles = ['Input x', "Target y", 'model f(x)', 'Inverse of f(x)', 'Inverse of y']
                images = [original, target, forward_output, inverse_forward, inverse_input]

                # print(original)
                # print(target)
                images = [img.transpose(1, 2, 0)  for img in images]
                images = [np.clip(img, 0, 1) for img in images]

                print(images[0].shape)

                for j in range(5):
                    plt.subplot(num_samples, 5, samples_visualized * 5 + j + 1)
                    plt.imshow(images[j])
                    plt.title(titles[j])
                    plt.axis('off')

                samples_visualized += 1

            if samples_visualized >= num_samples:
                break

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



# Example usage:
if __name__ == "__main__":
    train_loader, test_loader = load_cifar10_convert_data(batch_size=128, samples=30)
    # Print the shapes of the datasets
    for inputs, outputs in train_loader:
        print(f"inputs_train: {inputs.shape}")
        print(f"outputs_train: {outputs.shape}")
        break  # We only need to check one batch
