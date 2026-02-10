import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import grad

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the dataset and dataloader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageNet(root='./data', split='val', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def estimate_lipschitz(model, dataloader, num_samples=100):
    max_grad_norm = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        if i >= num_samples:
            break

        inputs = inputs.requires_grad_(True)  # Enable gradient computation for inputs
        outputs = model(inputs)

        # Compute the gradient of the output w.r.t. the input
        loss = outputs.norm()
        gradients = grad(outputs=loss, inputs=inputs, create_graph=True)[0]

        # Compute the norm of the gradients
        grad_norm = gradients.norm()

        # Update the maximum gradient norm
        if grad_norm.item() > max_grad_norm:
            max_grad_norm = grad_norm.item()

        print(f"Sample {i + 1}/{num_samples}: Grad Norm = {grad_norm.item()}")

    return max_grad_norm

# Estimate the Lipschitz constant
estimated_lipschitz = estimate_lipschitz(model, dataloader, num_samples=100)
print(f"Estimated Lipschitz constant: {estimated_lipschitz}")
