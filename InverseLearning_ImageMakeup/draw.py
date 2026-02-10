import matplotlib.pyplot as plt
import numpy as np

# Extracting the data manually from the provided image
layer_numbers = np.array([2, 4, 6, 8, 10, 12])

# Values extracted from the table in the image
# MSE values for each model (ResNet, iResNet, DipDNN, NICE)
mse_values = {
    "ResNet": [0.0000001, 0.0000001, 0.000080, 0.000395, 21.894579, 580.581360],
    "iResNet": [0.365727, 0.358352, 0.349940, 0.338674, 0.329622, 0.320366],
    "DipDNN": [1.6427931768703e-05, 4.2925599700538e-05, 4.672685349760286e-05, 1.2301887863941374e-07, 7.71113336683498e-06, 1.713329083941062e-06],
    "NICE": [0.084194, 0.000007, 0.000133, 0.002533, 0.002052, 0.000114]
}

# Inverse MSE values for each model
inverse_mse_values = {
    "ResNet": [14280628014940.16, 283828950.0, 88.15900421142578, 0.014294304884970188, np.nan, np.nan],
    "iResNet": [1.8664299524343e-05, 4.1382684022703e-05, 6.604623997763435e-05, 1.0097867281791e-05, 1.3544911271081805e-05, 1.7219414311653845e-05],
    "DipDNN": [3.1297477088264e-13, 6.553035024337017e-13, 8.294858723500997e-12, 4.837034328630807e-09, 5.829573410665034e-07, 5.518928542731924e-07],
    "NICE": [1.5324344322388246e-15, 2.112037902221261e-15, 2.5392666678805e-15, 4.001128685912191e-15, 4.019844725914722e-15, 1.9599935840145953e-15]
}

# True Inverse MSE values for each model
true_inv_mse_values = {
    "ResNet": [14281580960808.96, 28375402.0, 87.50181579589844, 0.3585062325000763, np.nan, np.nan],
    "iResNet": [0.37058356404304504, 0.3663056194782257, 0.3625808060169269, 0.35001271963119507, 0.3483770489692688, 0.34092676639556885],
    "DipDNN": [0.24236226081848145, 0.00385855267029702663, 0.0005478195380419493, 0.56412214040752693, 83278.9375, 30894190.0],
    "NICE": [59.884727478027344, 0.000650304777820585, 0.0004198458045721054, 0.06764118373394012, 328.589599609375, 3.456038331806213e-05]
}

# Create 3 separate plots for MSE, Inverse MSE, and True Inv MSE




# Plot 1: Log MSE values
plt.figure(figsize=(10, 6))
for model, values in mse_values.items():
    plt.plot(layer_numbers, np.log10(values), marker='o', label=model)
plt.xlabel('Layer Number')
plt.ylabel('Log(MSE)')
plt.title('Log(MSE) vs Layer Number')
plt.legend()
plt.grid(True)
# plt.show()

# Plot 2: Log Inverse MSE values
plt.figure(figsize=(10, 6))
for model, values in inverse_mse_values.items():
    plt.plot(layer_numbers, np.log10(values), marker='o', label=model)
plt.xlabel('Layer Number')
plt.ylabel('Log(Inverse MSE)')
plt.title('Log(Inverse MSE) vs Layer Number')
plt.legend()
plt.grid(True)
# plt.show()

# Plot 3: Log True Inverse MSE values
plt.figure(figsize=(10, 6))
for model, values in true_inv_mse_values.items():
    plt.plot(layer_numbers, np.log10(values), marker='o', label=model)
plt.xlabel('Layer Number')
plt.ylabel('Log(True Inverse MSE)')
plt.title('Log(True Inverse MSE) vs Layer Number')
plt.legend()
plt.grid(True)
# plt.show()
