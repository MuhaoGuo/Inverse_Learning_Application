from utils import *


class SmallNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SmallNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        # self.fc3 = nn.Linear(input_dim, input_dim)
        # self.activation = nn.ReLU()
        # self.activation = nn.Sigmoid()
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.activation(self.fc2(x))
        # x = self.fc3(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, alpha=0.9):
        super(ResidualBlock, self).__init__()
        self.alpha = alpha  # Scaling coefficient
        self.residual_function = SmallNetwork(input_dim)

    def forward(self, x):
        return x + self.alpha * self.residual_function(x)
        # return self.residual_function(x)

    def inverse(self, y):
        # Use fixed-point iteration to find the inverse
        x = y
        for _ in range(100):
            x = y - self.alpha * self.residual_function(x)  # y = x + af(x) --> x = y
        return x

# Define the i-ResNet
class ResNet(nn.Module):
    def __init__(self, input_dim, num_blocks):
        super(ResNet, self).__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(input_dim) for _ in range(num_blocks)])

    def forward(self, x):

        # x = x.reshape(x.shape[0], -1)  # (batchsize, -1)

        for block in self.blocks:
            x = block(x)

        # x = x.reshape(x.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
        return x

    def inverse(self, y):
        # y = y.reshape(y.shape[0], -1)  # (batchsize, -1)

        for block in reversed(self.blocks):
            y = block.inverse(y)

        # y = y.reshape(y.shape[0], 1, int(np.sqrt(input_dim)), int(np.sqrt(input_dim)))
        return y

    def count_parameters(self):
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
train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_data(device)


input_dim = 1
num_blocks = 3
learning_rate = 1e-3
epochs = 2000

model_name = "ResNet"
data_name = "Reacher"
model = ResNet(input_dim=input_dim, num_blocks=num_blocks).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

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
# Calculate MSE
fwd_error = mean_squared_error(y_train_tensor.cpu().detach().numpy(), train_pred.cpu().detach().numpy())
inv_fake_error = mean_squared_error(X_train_tensor.cpu().detach().numpy(), inverse_of_fx_train.cpu().detach().numpy())
inv_real_error = mean_squared_error(X_train_tensor.cpu().detach().numpy(), inverse_of_y_train.cpu().detach().numpy())
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
data = pd.read_csv("results/Truth_reacher_behavior.csv")
action = data["action"]
action = action.apply(lambda x: eval(x))
action = np.array(list(action.values))
action[:, [0]] = inverse_of_y

action_new = [str(list(obs)) for obs in action]
data["action"] = action_new
data.to_csv(f"{data_name}_{model_name}_real_inv.csv", index=False)



