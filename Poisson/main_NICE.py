from utils import *
class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, process_first_part=True):
        """
        Fix the bug that when dim == 1
        Affine Coupling Layer as per NICE architecture, now supports 1D input.

        Args:
            input_dim (int): Dimension of the input.
            process_first_part (bool): Whether to process the first or second part of the input.
        """
        super(AffineCouplingLayer, self).__init__()
        self.process_first_part = process_first_part
        self.input_dim = input_dim

        # Check for edge case where input_dim == 1
        if input_dim == 1:
            self.is_single_dim = True
            network_input_dim = input_dim
            network_output_dim = input_dim
        else:
            self.is_single_dim = False
            self.split_dim1 = input_dim // 2
            self.split_dim2 = input_dim - self.split_dim1

            if self.process_first_part:
                network_input_dim = self.split_dim1
                network_output_dim = self.split_dim2
            else:
                network_input_dim = self.split_dim2
                network_output_dim = self.split_dim1

        # Define the network
        self.network = nn.Sequential(
            nn.Linear(network_input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, network_output_dim)
        )

    def forward(self, x):
        """
        Forward pass of the Affine Coupling Layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Output tensor after coupling.
        """
        if self.is_single_dim:  # Handle 1D input
            shift = self.network(x)
            y = x + shift
            return y
        else:  # Handle regular multi-dimensional input
            x1, x2 = x.split([self.split_dim1, self.split_dim2], dim=1)
            if self.process_first_part:
                shift = self.network(x1)
                y1 = x1
                y2 = x2 + shift
            else:
                shift = self.network(x2)
                y1 = x1 + shift
                y2 = x2
            y = torch.cat((y1, y2), dim=1)
            return y

    def inverse(self, y):
        """
        Inverse pass of the Affine Coupling Layer.

        Args:
            y (Tensor): Output tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Reconstructed input tensor.
        """
        if self.is_single_dim:  # Handle 1D input
            shift = self.network(y)
            x = y - shift
            return x
        else:  # Handle regular multi-dimensional input
            y1, y2 = y.split([self.split_dim1, self.split_dim2], dim=1)
            if self.process_first_part:
                shift = self.network(y1)
                x1 = y1
                x2 = y2 - shift
            else:
                shift = self.network(y2)
                x1 = y1 - shift
                x2 = y2
            x = torch.cat((x1, x2), dim=1)
            return x


class NICEResNet(nn.Module):
    def __init__(self, input_dim, num_blocks):
        """
        Modified NICE model inspired by ResNet architecture.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Number of hidden units in the coupling layers.
            num_blocks (int): Number of Affine Coupling Layers (blocks) in the model.
        """
        super(NICEResNet, self).__init__()
        self.input_dim = input_dim
        self.num_blocks = num_blocks

        # Initialize Affine Coupling Layers
        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            process_first_part = (i % 2 == 0)  # Alternate processing parts for each block
            self.layers.append(AffineCouplingLayer(input_dim, process_first_part))

        # Learnable scaling factor
        self.scaling_factor = nn.Parameter(torch.ones(1, input_dim))

    def forward(self, x):
        """
        Forward pass through the NICEResNet model.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Transformed output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        x = x * self.scaling_factor
        return x

    def inverse(self, y):
        """
        Inverse pass through the NICEResNet model.

        Args:
            y (Tensor): Output tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Reconstructed input tensor.
        """
        y = y / self.scaling_factor
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_model(model, train_loader, optimizer, criterion, epochs, device=torch.device('cpu')):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, targets in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # model.apply_weight_masks()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / len(train_loader))


device = get_device()
print("device", device)
# Set the seed before initializing the model or any random operations
set_seed(42)  # You can use any seed value of your choice
train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_poisson_data_sampling(device)


input_dim = 513
num_blocks = 2
learning_rate = 1e-5
epochs = 2000

model_name = "NICE"
data_name = "Poisson"
model = NICEResNet(input_dim=input_dim, num_blocks=num_blocks).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
# criterion = nn.L1Loss()

### train model #########################
train_model(model, train_loader, optimizer, criterion, epochs, device)
model.eval()

with torch.no_grad():
    inverse_error = 0.0
    inverse_error_real = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass: input -> inverted
        outputs = model(inputs)

        # Inverse pass: f(x) -> input
        reconstructed_inputs = model.inverse(outputs)
        reconstructed_inputs_real = model.inverse(targets)
        # print("reconstructed_inputs", reconstructed_inputs)

        # Compute inverse error (MSE between original inputs and reconstructed inputs)
        loss = criterion(reconstructed_inputs, inputs)
        loss_real = criterion(reconstructed_inputs_real, inputs)

        inverse_error += loss.item() * inputs.size(0)
        inverse_error_real += loss_real.item() * inputs.size(0)

    avg_inverse_error = inverse_error / len(train_loader.dataset)
    avg_inverse_error_real = inverse_error_real / len(train_loader.dataset)

    print(f'Inverse Pass Error (MSE) after training: {avg_inverse_error}')
    print(f'Inverse Pass Error (MSE) after training real: {avg_inverse_error_real}')


### Test model on Train #########################

train_pred = model(X_train_tensor)
inverse_of_fx_train = model.inverse(train_pred)
inverse_of_y_train = model.inverse(y_train_tensor)



fig, axis = plt.subplots(1, 8, figsize=(20, 2))
Nx = 512
xmin, xmax, ymin, ymax = 0, 1, 0, 1
im0 = axis[0].imshow(y_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "True y (u)")
axis[0].set_title("True y")
im1 = axis[1].imshow(train_pred.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "Pred y")
axis[1].set_title("Pred y")
im2 = axis[2].imshow(y_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1) - train_pred.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "Fwd error")
axis[2].set_title("Fwd error")
im3 = axis[3].imshow(X_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "True X (f)")
axis[3].set_title("True X")
im4 = axis[4].imshow(inverse_of_fx_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "fake inv X (from fx)")
axis[4].set_title("fake inv X")
im5 = axis[5].imshow(inverse_of_y_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "real inv X (from y)")
axis[5].set_title("real inv X")
im6 = axis[6].imshow(inverse_of_fx_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1) -X_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1) , extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "fake inv error")
axis[6].set_title("fake inv error")
im7 = axis[7].imshow(inverse_of_y_train.cpu().detach().numpy().reshape(Nx + 1, Nx + 1)- X_train_tensor.cpu().detach().numpy().reshape(Nx + 1, Nx + 1), extent=[xmin, xmax, ymin, ymax], origin='lower',
                     aspect='auto',  label = "real inv error")
axis[7].set_title("real inv error")
plt.savefig(f"./results/{data_name}_{model_name}.png")


# Calculate MSE
fwd_error = mean_squared_error(y_train_tensor.cpu().detach().numpy(), train_pred.cpu().detach().numpy())
inv_fake_error = mean_squared_error(X_train_tensor.cpu().detach().numpy(), inverse_of_fx_train.cpu().detach().numpy())
inv_real_error = mean_squared_error(X_train_tensor.cpu().detach().numpy(), inverse_of_y_train.cpu().detach().numpy())
print("Training Data ..... ")
print(f"{model_name} MSE train fwd_error: {fwd_error}")
print(f"{model_name} MSE train inv_fake_error: {inv_fake_error}")
print(f"{model_name} MSE train inv_real_error: {inv_real_error}")
plt.show()






