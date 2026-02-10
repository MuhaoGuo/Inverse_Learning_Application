import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Define time variable
time = 60

# Construct filename
filename = f'noise_data_selected_{time}.csv'

# Read data from file
data = pd.read_csv(filename)

# Extract variables
x = data['x']
y = data['y']
uA = data['uA_error']

# Create grid for interpolation
numPoints = 100
grid_x = np.linspace(x.min(), x.max(), numPoints)
grid_y = np.linspace(y.min(), y.max(), numPoints)
gridX, gridY = np.meshgrid(grid_x, grid_y)

# Interpolate data
grid_uA = griddata((x, y), uA, (gridX, gridY), method='cubic')

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(gridX, gridY, grid_uA, cmap='summer')

# Customization
plt.colorbar(surf)
surf.set_clim(0, 2e-5)  # Adjust color axis limits

# Axis labels (uncomment if needed)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Error')

# View angle
ax.view_init(45, 45)

# Hide axis and grid lines for a cleaner look
ax.set_axis_off()

# Set figure background color to none for transparency
fig.patch.set_alpha(0)

# Save figure
plt.savefig('result_with_noise.png', transparent=True)

# Show plot
plt.show()
