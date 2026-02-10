import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import FuncFormatter

# Define time variable
time = 60
# model_name = "dipdnn"
# model_name = "resnet"
# model_name = "iresnet"
# model_name = "NICE"

# model_name = "conv_iresnet"
# model_name = "conv_resnet"
# model_name = "conv_resnet_kernel1"
# model_name = "conv_resnet_kernel2"
# model_name = "conv_iresnet_kernel1"
model_name = "conv_iresnet_kernel2"

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


# Define the custom formatter function
def custom_format(x, pos):
    if abs(x) < 1e-2 and x != 0:
        return f'{x:.1e}'
    else:
        return f'{x:.2f}'


def plot_3_metrix():
    # Create the figure and 1x5 subplots
    fig, axs = plt.subplots(3, 1, figsize=(5, 25/8*3), subplot_kw={'projection': '3d'})
    for i in range(0, 3):
        axs[i].grid(False)
        axs[i].xaxis.pane.set_alpha(0)  # 设置 X 轴的面板为透明
        axs[i].yaxis.pane.set_alpha(0)  # 设置 Y 轴的面板为透明
        axs[i].zaxis.pane.set_alpha(0)  # 设置 Z 轴的面板为透明
        axs[i].patch.set_alpha(0)
        axs[i].set_xlabel('x', fontsize=18)
        axs[i].set_ylabel('y', fontsize=18)
        # axs[i].set_zlabel('z', fontsize=18)

    surf3 = axs[0].plot_surface(gridX, gridY, grid_uA_pred, cmap='viridis')
    surf4 = axs[1].plot_surface(gridX, gridY, grid_wx_inv, cmap='summer')
    surf5 = axs[2].plot_surface(gridX, gridY, grid_wx_inv_real, cmap='summer')

    # Adjust layout for better spacing
    plt.tight_layout()

    for i in range(0, 3):
        xmin, xmax = axs[i].get_xlim()
        ymin, ymax = axs[i].get_ylim()
        zmin, zmax = axs[i].get_zlim()
        axs[i].set_xticks([xmin, xmax])
        axs[i].set_yticks([ymin, ymax])
        axs[i].set_zticks([zmin, zmax])

        axs[i].tick_params(axis='z', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='x', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='y', labelsize=15)  # Set x-tick font size

        # axs[i].zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].zaxis.set_major_formatter(FuncFormatter(custom_format))

        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig(f'./results/{model_name}_60.png', transparent=True)
    plt.show()
# plot_3_metrix()

def plot_2_metrix():
    # Create the figure and 1x5 subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 25/8*2), subplot_kw={'projection': '3d'})
    for i in range(0, 2):
        axs[i].grid(False)
        axs[i].xaxis.pane.set_alpha(0)  # 设置 X 轴的面板为透明
        axs[i].yaxis.pane.set_alpha(0)  # 设置 Y 轴的面板为透明
        axs[i].zaxis.pane.set_alpha(0)  # 设置 Z 轴的面板为透明
        axs[i].patch.set_alpha(0)
        axs[i].set_xlabel('x', fontsize=18)
        axs[i].set_ylabel('y', fontsize=18)
        # axs[i].set_zlabel('z', fontsize=18)

    surf3 = axs[0].plot_surface(gridX, gridY, grid_uA_pred, cmap='viridis')
    surf5 = axs[1].plot_surface(gridX, gridY, grid_wx_inv_real, cmap='summer')

    # Adjust layout for better spacing
    plt.tight_layout()

    for i in range(0,2):
        xmin, xmax = axs[i].get_xlim()
        ymin, ymax = axs[i].get_ylim()
        zmin, zmax = axs[i].get_zlim()
        axs[i].set_xticks([xmin, xmax])
        axs[i].set_yticks([ymin, ymax])
        axs[i].set_zticks([zmin, zmax])

        axs[i].tick_params(axis='z', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='x', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='y', labelsize=15)  # Set x-tick font size

        # axs[i].zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].zaxis.set_major_formatter(FuncFormatter(custom_format))

        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig(f'./results/{model_name}_60.png', transparent=True)
    plt.show()
# plot_2_metrix()

def plot_3_error_metrix():
    # Create the figure and 1x5 subplots
    fig, axs = plt.subplots(3, 1, figsize=(5, 25/8*3), subplot_kw={'projection': '3d'})
    for i in range(0, 3):
        axs[i].grid(False)
        axs[i].xaxis.pane.set_alpha(0)  # 设置 X 轴的面板为透明
        axs[i].yaxis.pane.set_alpha(0)  # 设置 Y 轴的面板为透明
        axs[i].zaxis.pane.set_alpha(0)  # 设置 Z 轴的面板为透明
        axs[i].patch.set_alpha(0)
        axs[i].set_xlabel('x', fontsize = 18)
        axs[i].set_ylabel('y', fontsize = 18)
        # axs[i].set_zlabel('z', fontsize= 18)

    surf6 = axs[0].plot_surface(gridX, gridY, grid_uA_error, cmap='autumn')
    surf7 = axs[1].plot_surface(gridX, gridY, grid_wx_error, cmap='autumn')
    surf8 = axs[2].plot_surface(gridX, gridY, grid_wx_error_real, cmap='autumn')
    plt.tight_layout()

    for i in range(0, 3):
        xmin, xmax = axs[i].get_xlim()
        ymin, ymax = axs[i].get_ylim()
        zmin, zmax = axs[i].get_zlim()
        axs[i].set_xticks([xmin, xmax])
        axs[i].set_yticks([ymin, ymax])
        axs[i].set_zticks([zmin, zmax])

        axs[i].tick_params(axis='z', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='x', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='y', labelsize=15)  # Set x-tick font size

        # axs[i].zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].zaxis.set_major_formatter(FuncFormatter(custom_format))

        # formatter = ScalarFormatter()
        # formatter.set_powerlimits((-3, 3))  # Forces scientific notation for numbers outside this range
        # axs[i].zaxis.set_major_formatter(formatter)

        # axs[i].zaxis.set_major_formatter(CustomLogFormatterSciNotation())

        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig(f'./results/{model_name}_60.png', transparent=True)
    plt.show()


def plot_2_error_metrix():
    # Create the figure and 1x5 subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 25/8*2), subplot_kw={'projection': '3d'})
    for i in range(0, 2):
        axs[i].grid(False)
        axs[i].xaxis.pane.set_alpha(0)  # 设置 X 轴的面板为透明
        axs[i].yaxis.pane.set_alpha(0)  # 设置 Y 轴的面板为透明
        axs[i].zaxis.pane.set_alpha(0)  # 设置 Z 轴的面板为透明
        axs[i].patch.set_alpha(0)
        axs[i].set_xlabel('x', fontsize = 18)
        axs[i].set_ylabel('y', fontsize = 18)
        # axs[i].set_zlabel('z', fontsize= 18)

    axs[0].plot_surface(gridX, gridY, grid_uA_error, cmap='autumn')
    axs[1].plot_surface(gridX, gridY, grid_wx_error_real, cmap='autumn')
    plt.tight_layout()

    for i in range(0, 2):
        xmin, xmax = axs[i].get_xlim()
        ymin, ymax = axs[i].get_ylim()
        zmin, zmax = axs[i].get_zlim()
        axs[i].set_xticks([xmin, xmax])
        axs[i].set_yticks([ymin, ymax])
        axs[i].set_zticks([zmin, zmax])

        axs[i].tick_params(axis='z', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='x', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='y', labelsize=15)  # Set x-tick font size

        # axs[i].zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].zaxis.set_major_formatter(FuncFormatter(custom_format))

        # formatter = ScalarFormatter()
        # formatter.set_powerlimits((-3, 3))  # Forces scientific notation for numbers outside this range
        # axs[i].zaxis.set_major_formatter(formatter)

        # axs[i].zaxis.set_major_formatter(CustomLogFormatterSciNotation())

        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig(f'./results/{model_name}_60.png', transparent=True)
    plt.show()
# plot_2_error_metrix()

def plot_8_all_metrix():
    # Create the figure and 1x5 subplots
    fig, axs = plt.subplots(8, 1, figsize=(5, 25), subplot_kw={'projection': '3d'})
    for i in range(0, 8):
        axs[i].grid(False)
        axs[i].xaxis.pane.set_alpha(0)  # 设置 X 轴的面板为透明
        axs[i].yaxis.pane.set_alpha(0)  # 设置 Y 轴的面板为透明
        axs[i].zaxis.pane.set_alpha(0)  # 设置 Z 轴的面板为透明
        axs[i].patch.set_alpha(0)
        axs[i].set_xlabel('x', fontsize = 18)
        axs[i].set_ylabel('y', fontsize = 18)
        axs[i].set_zlabel('z', fontsize= 18)

    # 1st subplot: x, y, wx (3D surface plot)
    surf1 = axs[0].plot_surface(gridX, gridY, grid_wx, cmap='summer')
    # axs[0].set_zlabel('wx')
    # axs[0].set_title('x, y, wx')
    # cbar1 = fig.colorbar(surf1, ax=axs[0], shrink=0.5, location='left', fraction=0.05, pad=0.1)
    # cbar1.set_ticks([np.min(grid_wx), np.max(grid_wx)])  # Set only min and max ticks
    # cbar1.ax.tick_params(labelsize=16)  # Change font size of colorbar ticks

    # 2nd subplot: x, y, uA_true (3D surface plot)
    surf2 = axs[1].plot_surface(gridX, gridY, grid_uA, cmap='viridis')
    # axs[1].set_zlabel('uA')
    # axs[1].set_title('x, y, uA_true')
    # fig.colorbar(surf2, ax=axs[1], shrink=0.5, location='left', fraction=0.05, pad=0.1)


    # 3rd subplot: x, y, uA_pred (3D surface plot)
    surf3 = axs[2].plot_surface(gridX, gridY, grid_uA_pred, cmap='viridis')
    # axs[2].set_zlabel('uA fwd')
    # axs[2].set_title('x, y, uA_pred')
    # fig.colorbar(surf3, ax=axs[2], shrink=0.5, location='left', fraction=0.05, pad=0.1)

    # 4th subplot: x, y, wx_inv (3D surface plot)
    surf4 = axs[3].plot_surface(gridX, gridY, grid_wx_inv, cmap='summer')
    # axs[3].set_zlabel('wx inverse')
    # axs[3].set_title('x, y, wx_inv')
    # axs[3].patch.set_alpha(0)
    # fig.colorbar(surf4, ax=axs[3], shrink=0.5, location='left', fraction=0.05, pad=0.1)

    # 5th subplot: x, y, wx_inv_real (3D surface plot)
    surf5 = axs[4].plot_surface(gridX, gridY, grid_wx_inv_real, cmap='summer')
    # axs[4].set_zlabel('wx inv-consist')
    # axs[4].set_title('x, y, wx_inv_real')
    # axs[4].patch.set_alpha(0)
    # fig.colorbar(surf5, ax=axs[4], shrink=0.5, location='left', fraction=0.05, pad=0.1)


    surf6 = axs[5].plot_surface(gridX, gridY, grid_uA_error, cmap='autumn')
    # axs[5].set_zlabel('uA fwd error')
    # axs[5].set_title('x, y, uA_error')
    # axs[5].patch.set_alpha(0)
    # fig.colorbar(surf6, ax=axs[5], shrink=0.5, location='left', fraction=0.05, pad=0.1)

    surf7 = axs[6].plot_surface(gridX, gridY, grid_wx_error, cmap='autumn')
    # axs[6].set_zlabel('wx inv error')
    # axs[6].set_title('x, y, wx_error')
    # axs[6].patch.set_alpha(0)
    # fig.colorbar(surf7, ax=axs[6], shrink=0.5, location='left', fraction=0.05, pad=0.1)

    surf8 = axs[7].plot_surface(gridX, gridY, grid_wx_error_real, cmap='autumn')
    # axs[7].set_zlabel('wx inv-consist error')
    # axs[7].set_title('x, y, wx_error_real')
    # axs[7].patch.set_alpha(0)
    # fig.colorbar(surf8, ax=axs[7], shrink=0.5, location='left', fraction=0.05, pad=0.1)

    # Adjust layout for better spacing
    plt.tight_layout()

    for i in range(0, 8):
        xmin, xmax = axs[i].get_xlim()
        ymin, ymax = axs[i].get_ylim()
        zmin, zmax = axs[i].get_zlim()
        axs[i].set_xticks([xmin, xmax])
        axs[i].set_yticks([ymin, ymax])
        axs[i].set_zticks([zmin, zmax])

        axs[i].tick_params(axis='z', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='x', labelsize=15)  # Set x-tick font size
        axs[i].tick_params(axis='y', labelsize=15)  # Set x-tick font size

        axs[i].zaxis.set_major_formatter(FuncFormatter(custom_format))

        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.savefig(f'./results/{model_name}_60.png', transparent=True)
    plt.show()

plot_8_all_metrix()