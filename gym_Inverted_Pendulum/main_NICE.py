from gym_Inverted_Pendulum.utils import *

# class AffineCouplingLayer(nn.Module):
#     def __init__(self, input_dim, process_first_part=True):
#         """
#         Affine Coupling Layer as per NICE architecture.
#
#         Args:
#             input_dim (int): Dimension of the input.
#             process_first_part (bool): Whether to process the first or second part of the input.
#         """
#         super(AffineCouplingLayer, self).__init__()
#         self.process_first_part = process_first_part
#         self.input_dim = input_dim
#         self.split_dim1 = input_dim // 2
#         self.split_dim2 = input_dim - self.split_dim1
#
#         if self.process_first_part:
#             network_input_dim = self.split_dim1
#             network_output_dim = self.split_dim2
#         else:
#             network_input_dim = self.split_dim2
#             network_output_dim = self.split_dim1
#
#         self.network = nn.Sequential(
#             nn.Linear(network_input_dim, input_dim),
#             nn.ReLU(),
#             nn.Linear(input_dim, input_dim),
#             nn.ReLU(),
#             nn.Linear(input_dim, network_output_dim)
#         )
#
#     def forward(self, x):
#         """
#         Forward pass of the Affine Coupling Layer.
#
#         Args:
#             x (Tensor): Input tensor of shape [batch_size, input_dim].
#
#         Returns:
#             Tensor: Output tensor after coupling.
#         """
#         x1, x2 = x.split([self.split_dim1, self.split_dim2], dim=1)
#         if self.process_first_part:
#             shift = self.network(x1)
#             y1 = x1
#             y2 = x2 + shift
#         else:
#             shift = self.network(x2)
#             y1 = x1 + shift
#             y2 = x2
#         y = torch.cat((y1, y2), dim=1)
#         return y
#
#     def inverse(self, y):
#         """
#         Inverse pass of the Affine Coupling Layer.
#
#         Args:
#             y (Tensor): Output tensor of shape [batch_size, input_dim].
#
#         Returns:
#             Tensor: Reconstructed input tensor.
#         """
#         y1, y2 = y.split([self.split_dim1, self.split_dim2], dim=1)
#         if self.process_first_part:
#             shift = self.network(y1)
#             x1 = y1
#             x2 = y2 - shift
#         else:
#             shift = self.network(y2)
#             x1 = y1 - shift
#             x2 = y2
#         x = torch.cat((x1, x2), dim=1)
#         return x
#

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
train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_InvertedPendulum_data(device)


input_dim = 4
num_blocks = 2
learning_rate = 1e-2
epochs = 2000

model_name = "NICE"
data_name = "InvertedPendulum"
model = NICEResNet(input_dim=input_dim, num_blocks=num_blocks).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

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

fig, axes = plt.subplots(4, 1, figsize=(10, 8))
plt.title(model_name)
axes[0].plot(y_train_tensor.cpu().detach().numpy()[:, [0]], label = "True y")
axes[1].plot(train_pred.cpu().detach().numpy()[:, [0]], label = "Pred y")
axes[2].plot(X_train_tensor.cpu().detach().numpy()[:, [0]], label = "True X")
axes[3].plot(inverse_of_y_train.cpu().detach().numpy()[:, [0]], label = "Inverse X")
axes[0].legend()
axes[1].legend()
axes[2].legend()
axes[3].legend()
plt.show()

# Calculate MSE
fwd_error = mean_squared_error(y_train_tensor.cpu().detach().numpy()[:, [0]], train_pred.cpu().detach().numpy()[:, [0]])
inv_fake_error = mean_squared_error(X_train_tensor.cpu().detach().numpy()[:, [0]], inverse_of_fx_train.cpu().detach().numpy()[:, [0]])
inv_real_error = mean_squared_error(X_train_tensor.cpu().detach().numpy()[:, [0]], inverse_of_y_train.cpu().detach().numpy()[:, [0]])
print("Training Data ..... ")
print(f"{model_name} MSE train fwd_error: {fwd_error}")
print(f"{model_name} MSE train inv_fake_error: {inv_fake_error}")
print(f"{model_name} MSE train inv_real_error: {inv_real_error}")
#
# ### Test model #########################
# test_pred = model(X_test_tensor)
# inverse_of_fx_test = model.inverse(test_pred)
# inverse_of_y_test = model.inverse(y_test_tensor)
# # Calculate MSE
# fwd_error = mean_squared_error(y_test_tensor.cpu().detach().numpy(), test_pred.cpu().detach().numpy())
# inv_fake_error = mean_squared_error(X_test_tensor.cpu().detach().numpy(), inverse_of_fx_test.cpu().detach().numpy())
# inv_real_error = mean_squared_error(X_test_tensor.cpu().detach().numpy(), inverse_of_y_test.cpu().detach().numpy())
# print("Test Data ..... ")
# print(f"{model_name} MSE test fwd_error: {fwd_error}")
# print(f"{model_name} MSE test inv_fake_error: {inv_fake_error}")
# print(f"{model_name} MSE test inv_real_error: {inv_real_error}")

# replace the action with learned values
# inverse_of_y = np.concatenate((inverse_of_y_train.cpu().detach().numpy(), inverse_of_y_test.cpu().detach().numpy()), axis=0)
inverse_of_y = inverse_of_y_train.cpu().detach().numpy()
data = pd.read_csv("results/InvertedPendulum_truth.csv")
action = data["action"]
action = action.apply(lambda x: eval(x))
action = np.array(list(action.values))
action[:, [0]] = inverse_of_y[:, [0]]

action_new = [str(list(obs)) for obs in action]
data["action"] = action_new
data.to_csv(f"{data_name}_{model_name}_real_inv.csv", index=False)




