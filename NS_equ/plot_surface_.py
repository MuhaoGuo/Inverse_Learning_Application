import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Define time variable
time = 60
# model_name = "dipdnn"
# model_name = "resnet"
# model_name = "iresnet"
# model_name = "NICE"


# model_name = "conv_resnet_kernel1"
# model_name = "conv_resnet_kernel2"
# model_name = "conv_resnet"
#
# model_name = "conv_iresnet_kernel1"
# model_name = "conv_iresnet_kernel2"
# model_name = "conv_iresnet"


filename = f'./results/{model_name}_data_selected_{time}.csv'
df = pd.read_csv(filename)

# Extract the data
wx = df['wx'].values
wy = df['wy'].values
x = df['x'].values
y = df['y'].values
uA_true = df['uA_true'].values
uA_pred = df['uA_pred'].values
wx_inv = df['wx_inv'].values
wy_inv = df['wy_inv'].values
wx_inv_real = df['wx_inv_real'].values
wy_inv_real = df['wy_inv_real'].values

uA_error = df['uA_error'].values
wx_error = df['wx_error'].values
wx_error_real = df['wx_error_real'].values

print(f"{model_name} Forward MSE:", np.mean(abs(uA_error)))
print(f"{model_name} Inv MSE:", np.mean(abs(wx_error)))
print(f"{model_name} Inv real MSE:", np.mean(abs(wx_error_real)))

# exit()

# Create grid for interpolation
numPoints = 100
grid_x = np.linspace(x.min(), x.max(), numPoints)
grid_y = np.linspace(y.min(), y.max(), numPoints)
gridX, gridY = np.meshgrid(grid_x, grid_y)

# Interpolate data
grid_uA = griddata((x, y), uA_true, (gridX, gridY), method='cubic')
grid_uA_pred = griddata((x, y), uA_pred, (gridX, gridY), method='cubic')
grid_wx = griddata((x, y), wx, (gridX, gridY), method='cubic')
grid_wx_inv = griddata((x, y), wx_inv, (gridX, gridY), method='cubic')
grid_wx_inv_real = griddata((x, y), wx_inv_real, (gridX, gridY), method='cubic')

grid_uA_error = griddata((x, y), uA_error, (gridX, gridY), method='cubic')
grid_wx_error = griddata((x, y), wx_error, (gridX, gridY), method='cubic')
grid_wx_error_real = griddata((x, y), wx_error_real, (gridX, gridY), method='cubic')

# Create the figure and 1x5 subplots
fig, axs = plt.subplots(1, 8, figsize=(25, 5), subplot_kw={'projection': '3d'})

# 1st subplot: x, y, wx (3D surface plot)
surf1 = axs[0].plot_surface(gridX, gridY, grid_wx, cmap='summer')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_zlabel('wx')
axs[0].set_title('x, y, wx')
fig.colorbar(surf1, ax=axs[0], shrink=0.5, aspect=5)

# 2nd subplot: x, y, uA_true (3D surface plot)
surf2 = axs[1].plot_surface(gridX, gridY, grid_uA, cmap='viridis')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_zlabel('uA_true')
axs[1].set_title('x, y, uA_true')
fig.colorbar(surf2, ax=axs[1], shrink=0.5, aspect=5)

# 3rd subplot: x, y, uA_pred (3D surface plot)
surf3 = axs[2].plot_surface(gridX, gridY, grid_uA_pred, cmap='viridis')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].set_zlabel('uA_pred')
axs[2].set_title('x, y, uA_pred')
fig.colorbar(surf3, ax=axs[2], shrink=0.5, aspect=5)

# 4th subplot: x, y, wx_inv (3D surface plot)
surf4 = axs[3].plot_surface(gridX, gridY, grid_wx_inv, cmap='summer')
axs[3].set_xlabel('x')
axs[3].set_ylabel('y')
axs[3].set_zlabel('wx_inv')
axs[3].set_title('x, y, wx_inv')
fig.colorbar(surf4, ax=axs[3], shrink=0.5, aspect=5)

# 5th subplot: x, y, wx_inv_real (3D surface plot)
surf5 = axs[4].plot_surface(gridX, gridY, grid_wx_inv_real, cmap='summer')
axs[4].set_xlabel('x')
axs[4].set_ylabel('y')
axs[4].set_zlabel('wx_inv_real')
axs[4].set_title('x, y, wx_inv_real')
fig.colorbar(surf5, ax=axs[4], shrink=0.5, aspect=5)


surf6 = axs[5].plot_surface(gridX, gridY, grid_uA_error, cmap='summer')
axs[5].set_xlabel('x')
axs[5].set_ylabel('y')
axs[5].set_zlabel('uA_error')
axs[5].set_title('x, y, uA_error')
fig.colorbar(surf6, ax=axs[5], shrink=0.5, aspect=5)

surf7 = axs[6].plot_surface(gridX, gridY, grid_wx_error, cmap='summer')
axs[6].set_xlabel('x')
axs[6].set_ylabel('y')
axs[6].set_zlabel('wx_error')
axs[6].set_title('x, y, wx_error')
fig.colorbar(surf7, ax=axs[6], shrink=0.5, aspect=5)

surf8 = axs[7].plot_surface(gridX, gridY, grid_wx_error_real, cmap='summer')
axs[7].set_xlabel('x')
axs[7].set_ylabel('y')
axs[7].set_zlabel('wx_error_real')
axs[7].set_title('x, y, wx_error_real')
fig.colorbar(surf8, ax=axs[7], shrink=0.5, aspect=5)

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig(f'./results/{model_name}_60.png', transparent=True)
plt.show()

exit()








# Plotting
fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(gridX, gridY, grid_uA, cmap='summer')
plt.colorbar(surf)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('uA')
# surf.set_clim(0, 2e-5)  # Adjust color axis limits

ax1 = fig.add_subplot(122, projection='3d')
surf1 = ax1.plot_surface(gridX, gridY, grid_uA_pred, cmap='summer')
plt.colorbar(surf1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('uA pred')
# Axis labels (uncomment if needed)

# View angle
ax.view_init(45, 45)

# Hide axis and grid lines for a cleaner look
# ax.set_axis_off()

# Save figure
plt.savefig('result_without_noise.png', transparent=True)

# Show plot
plt.show()
