import matplotlib.pyplot as plt
import numpy as np

# Manually extracting values from the image for each row

# Number of layers (x-axis)
layers = np.array([2, 4, 6, 8, 10, 12])

### forward mse ######################################
# ResNet accuracy values
resnet_acc = np.array([0.0005835551419295371, 0.0002812569728121615, 0.00019949450506828725,
                       0.00014142406871542335, 5.106455864734016e-05, 3.9873822970548645e-05])

# iResNet accuracy values
iresnet_acc = np.array([0.050531331449747086, 0.0378844179213047, 0.02544865384697914,
                        0.012209424749016762, 0.0024811699986457825, 0.00016344636969733983])

# DipDNN accuracy values
dipdnn_acc = np.array([0.0013728757621720433, 0.0006398205296136439, 0.00031139873317442834,
                       0.00012377198436297476, 0.0001826223888201639, 0.0005454652127809823])

# DipDNN accuracy values
nice_acc = np.array([0.35405975580215454,0.0021051198709756136, 0.0011913699563592672, 0.001509023248218
                    ,0.0011289413087069988, 0.0012389475014060736])


fig_name = "layers vs forward MSE"


### fake inverse ######################################
# # ResNet Inverse Accuracy values
# resnet_acc = np.array([4.738352206367921e+72, 5.542998362057663e+63, 4.857140059115673e+53, 
#                            4.550505626220262e+44, 1.010084549745347e+47, 5.794434458494220e+00])

# # iResNet Inverse Accuracy values
# iresnet_acc = np.array([3.940167836588451e-15, 1.1898150657826161e-14, 2.911885216663637e-14,
#                             5.700597201768887e-14, 8.910234984487310e-14, 3.0949548419309076e-13])

# # DipDNN Inverse Accuracy values
# dipdnn_acc = np.array([1.5987417280349617e-11, 1.1788836700162568e-11, 1.8182592523732083e-11,
#                            4.838236049080365e-10, 1.6642531068916605e-10, 1.5756829456554985e-09])

# fig_name = "layers vs inverse true MSE"
 ###################################### ######################################

# ## True inverse #######################################
# # ResNet True Inverse Accuracy values
# resnet_acc = np.array([4.745026970833846e+72, 8.494762119668577e+77, 7.259645884677072e+72, 
#                                 4.3664425368581776e+76, 1.3745010018583574e+74, 8.866336783511062e+00])

# # iResNet True Inverse Accuracy values
# iresnet_acc = np.array([0.25805757572407206, 0.22809722699808882, 0.11447653068722662,
#                                  0.10973170102299103, 0.09457730388502521, 0.04307042371130809])

# # DipDNN True Inverse Accuracy values
# dipdnn_acc = np.array([10.175239311160363, 4.133056961845552, 3.346880521136519,
#                                 52.85989167790606, 325.04281212613944, 13520.902440897999])


# fig_name = "layers vs inverse true MSE"

# Plotting the data
plt.figure(figsize=(8, 6))
plt.plot(layers, resnet_acc, label='ResNet', marker='o')
plt.plot(layers, iresnet_acc, label='iResNet', marker='o')
plt.plot(layers, dipdnn_acc, label='DipDNN', marker='o')
plt.plot(layers, nice_acc, label='NICE', marker='o')

# Setting logarithmic scale for y-axis
plt.yscale('log')

# Labeling the axes
plt.xlabel('Number of Layers')
plt.ylabel('Accuracy (log scale)')

# Adding title and legend
plt.title('MSE vs Number of Layers for Different Models')
plt.legend()
plt.savefig(f"./results/{fig_name}")

# Displaying the plot
plt.show()
