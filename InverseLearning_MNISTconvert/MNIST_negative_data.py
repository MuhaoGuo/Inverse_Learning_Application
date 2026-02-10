import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# Define different transformation functions
def invert_colors(image):
    """Invert black/white (color inversion)"""
    return transforms.functional.invert(image)

def invert_contrast(image):
    """Invert image contrast"""
    # For each pixel  p' = 255−p, simple flipping of pixel intensities.
    image_np = np.array(image)
    inverted_contrast = 255 - image_np  # Invert pixel values
    return torch.from_numpy(inverted_contrast).to(torch.uint8)

def invert_with_noise(image, noise_level=20):
    """Invert image intensities with added Gaussian noise"""
    # p' = 255−p+noise,
    image_np = np.array(image)
    inverted_image = 255 - image_np  # Invert pixel values
    noise = np.random.normal(0, noise_level, image_np.shape)
    noisy_inverted = np.clip(inverted_image + noise, 0, 255)
    return torch.from_numpy(noisy_inverted).to(torch.uint8)

def inverse_scale(image):
    """Scale pixel intensity inversely (low intensity becomes high)"""
    # p'  = 255⋅(1−p/255) y=1−x after normalization.
    image_np = np.array(image) / 255.0  # Normalize to [0, 1]
    inverse_scaled = 1 - image_np  # Apply inverse scaling
    inverse_scaled = (inverse_scaled * 255).astype(np.uint8)  # Rescale
    return torch.from_numpy(inverse_scaled)

# Define inverse scaling (intensity scaling)
def inverse_scale(image):
    """Inverse proportionality by scaling intensities"""
    image_np = np.array(image).astype(np.float32)
    # Handle zeros in input (to avoid division by zero)
    image_np[image_np == 0] = 1  # Replace 0s with 1 to prevent division by zero
    inverse_scaled_image = 255 / image_np  # y = 255 / x
    inverse_scaled_image = np.clip(inverse_scaled_image, 0, 255)  # Ensure values are in [0, 255]
    return torch.from_numpy(inverse_scaled_image).to(torch.uint8)

# Define geometric and intensity-based inverse transformation
def geometric_and_intensity_inversion(image):
    """Flip and invert intensities, creating inverse proportional relationship"""
    # Flip the image horizontally
    flipped_image = transforms.functional.hflip(image)
    # Apply intensity inversion: y = 255 - flipped_image
    flipped_image_np = np.array(flipped_image)
    inverse_intensity_image = 255 - flipped_image_np
    return torch.from_numpy(inverse_intensity_image).to(torch.uint8)

# Dataset generator function
def generate_dataset(transform_type='invert_colors', data=None, direction='horizontal', noise_level=20):
    """Generate datasets with chosen transformation applied"""
    transform_functions = {
        'invert_colors': invert_colors,
        'invert_contrast': invert_contrast,
        'invert_with_noise': lambda img: invert_with_noise(img, noise_level=noise_level),
        'inverse_scale': inverse_scale,
        'geometric_and_intensity_inversion': geometric_and_intensity_inversion
    }
    transform_fn = transform_functions.get(transform_type)

    if transform_fn is None:
        raise ValueError(f"Invalid transform type: {transform_type}")

    # Apply transformation to the entire dataset
    transformed_dataset = [(image, transform_fn(image)) for image, label in data]

    return transformed_dataset



# Visualize some examples from the generated dataset
def visualize_transformed_examples(dataset, num_examples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_examples):
        original_image, transformed_image = dataset[i]
        plt.subplot(2, num_examples, i+1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_examples, i+1+num_examples)
        plt.imshow(transformed_image, cmap='gray')
        plt.title('Transformed')
        plt.axis('off')
        plt.savefig("./results/raw.png")
    # plt.show()



# Example to use dataset as NN input-output pairs
def prepare_for_nn(dataset):
    """Prepare the dataset in the format required for NN (inputs and outputs as tensors)"""
    inputs = []
    outputs = []
    for original_image, transformed_image in dataset:
        input_tensor = transforms.ToTensor()(original_image)  # Convert to tensor
        output_tensor = transforms.ToTensor()(transformed_image)
        inputs.append(input_tensor)
        outputs.append(output_tensor)

    inputs = torch.stack(inputs)  # Stack into batch
    outputs = torch.stack(outputs)
    return inputs, outputs



if __name__ == "__main__":
    # Load MNIST dataset
    
    data_path = '../data/'
    mnist_data_train = torchvision.datasets.MNIST(root=data_path, train=True, download=True,)
    mnist_data_test = torchvision.datasets.MNIST(root=data_path, train=False, download=True,)

    # Example: Generate dataset with color inversion
    transformed_dataset_train = generate_dataset(transform_type='invert_colors', data= mnist_data_train)
    transformed_dataset_test = generate_dataset(transform_type='invert_colors', data=mnist_data_test)

    # Visualize the transformed examples (color inversion in this case)
    visualize_transformed_examples(transformed_dataset_train)
    visualize_transformed_examples(transformed_dataset_test)

    # Prepare the dataset for training the neural network
    inputs_train, outputs_train = prepare_for_nn(transformed_dataset_train)
    inputs_test, outputs_test = prepare_for_nn(transformed_dataset_test)
    # print(inputs_train[0, 0, 0])
    # print(outputs_train[0, 0, 0])

    # inputs and outputs are now ready to be fed into a neural network
    print(f"inputs_train: {inputs_train.shape}")
    print(f"outputs_train: {outputs_train.shape}")

    print(f"inputs_test: {inputs_test.shape}")
    print(f"outputs_test: {outputs_test.shape}")


    # Compute the average pixel value for each image
    inputs_avg = inputs_test.mean(dim=(1, 2, 3)).cpu().numpy()  # Shape: (1000,)
    outputs_avg = outputs_test.mean(dim=(1, 2, 3)).cpu().numpy()  # Shape: (1000,)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(inputs_avg, outputs_avg, alpha=0.5)
    plt.title('Relationship Between Inputs and Outputs Averages')
    plt.xlabel('Input Average')
    plt.ylabel('Output Average')
    # # Optionally, add a line of best fit
    # m, b = np.polyfit(inputs_avg, outputs_avg, 1)
    # plt.plot(inputs_avg, m*inputs_avg + b, color='red', linewidth=2)
    plt.grid(True)
    plt.savefig("./results/raw_data.png")


    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(inputs_avg, alpha=0.5, label = "input")
    plt.plot(outputs_avg, alpha=0.5, label = "output")
    plt.title('Relationship Between index vs Inputs/Outputs Averages')
    plt.xlabel('index')
    plt.ylabel('Output Average')
    plt.legend()
    plt.grid(True)
    plt.savefig("./results/raw_data_index_values.png")


