#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 21:41:58 2023

@author: yuanjingyi
"""


from torch import Tensor
from torch.nn import Linear, MSELoss, functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable
from torch.utils.data import DataLoader
import timeit
from INN_data_synthetic import *
from utilis import *
from INN_model_layeroutputs import *
import argparse
import json

# # Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--num_blocks", type=int, required=True)
# args = parser.parse_args()

# # Use the learning rate from the command-line arguments
# num_blocks = args.num_blocks

num_blocks = 2
INN = Inverse_DNN(input_size, output_size, num_blocks)
print(dict(INN.named_parameters()))
criterion = MSELoss()
optimizer_INN = Adam(INN.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-05, amsgrad=False)

# define the number of epochs and the data set size
nb_epochs = 1001
batchsize = 100
mask_tril = torch.tril(torch.ones_like((INN.model[0].fc1.weight)))
mask_triu = torch.triu(torch.ones_like((INN.model[0].fc2.weight)))

start = timeit.default_timer()
reg_count = 0
for epoch in range(nb_epochs):
    epoch_loss = 0;
    for ix in range(int(n_train/batchsize)):
        batchinput = X_tensor[100*ix:100*(ix+1),:]
        batchoutput = Y_tensor[100*ix:100*(ix+1),:]
        y_pred, _ = INN.forward(Variable(batchinput))

        batch_loss = criterion(y_pred, Variable(batchoutput))

        epoch_loss = epoch_loss + batch_loss.data.item()

        optimizer_INN.zero_grad()

        batch_loss.backward()

        optimizer_INN.step()

        with torch.no_grad():
            for name, param in INN.named_parameters():
                    if 'weight' in name:
                        # Get the diagonal elements
                        diag_elements = torch.diag(param)
                        # Define threshold
                        threshold = 1e-2
                        # Find diagonal elements close to zero
                        mask = torch.abs(diag_elements) < threshold
                        # Specify a non-zero value to replace close-to-zero values
                        non_zero_value = 1e-1
                        # If there are elements close to zero, replace them with non_zero_value
                        if torch.any(mask):
                            param[torch.eye(param.size(0)).bool()] = torch.where(mask, torch.tensor(non_zero_value), diag_elements)
                            reg_count = reg_count +1
                            # print("Modified weight matrix:")
                            # print(name,param)

            for i in range(num_blocks):
                INN.model[i].fc1.weight.mul_(mask_tril)
                INN.model[i].fc2.weight.mul_(mask_triu)
    if epoch % 100 == 0:
        print("Epoch: {} Loss: {}".format(epoch, epoch_loss/(int(n_train/batchsize))))
stop = timeit.default_timer()
test_pred, _ = INN.forward(Variable(X_test_tensor))
train_pred, _ = INN.forward(Variable(X_tensor))
train_loss = np.mean((np.array(Y) - np.array(train_pred.detach().numpy()))** 2)
test_loss = np.mean((np.array(Y_test) - np.array(test_pred.detach().numpy()))** 2)
Train_MAPE = mean_absolute_percentage_error(Y,  train_pred.detach().numpy())
Test_MAPE = mean_absolute_percentage_error(Y_test,  test_pred.detach().numpy())
print(dict(INN.named_parameters()))
print("# of weights regularization", reg_count)
print("TrainingTime: ", stop - start)
print("The forward:\n MSE " + str(train_loss),test_loss)
print(" MAPE " + "{:.4f}%".format(Train_MAPE),"{:.4f}%".format(Test_MAPE) )


Y_est, layer_fwd_train  = INN.forward(X_tensor)
X_est, layer_inv_train = INN.inverse(Y_tensor)
X_reconstr, layer_inv_reconstr = INN.inverse(Y_est)
yx_err = np.mean((X_est.detach().numpy() - X_tensor.detach().numpy())** 2) # inverse estimation dependent on fwd error
xyx_err = np.mean((X_reconstr.detach().numpy() - X_tensor.detach().numpy())** 2) # reconstruction err independent on fwd err
print("The Inverse (MSE):\n one-direction learning " + str(yx_err), "\n one-to-one learning " + str(xyx_err))

print("One-to-one performance by block:")
for layer in range(num_blocks):
    ## propagation err of block inverse
    fwd_input = layer_fwd_train[-1-(layer+1)] #forward layer inputs
    inv_reconstr = layer_inv_reconstr[layer+1]
    zyz_err = np.mean((inv_reconstr.detach().numpy() - fwd_input.detach().numpy())** 2) #reconstruction of z in the middle layers
#    print("Block " + str( layer)+ " propagation" , zyz_err)
    ## ablation err of isolated block inverse
    fwd_output1 = layer_fwd_train[-1-(layer)]#forward layer outputs
    fwd_output2 = INN.model[-1-(layer)].forward(fwd_input)  #forward layer outputs
    if (fwd_output1 == fwd_output2).all():# check --> should be the same
        inv_output =  INN.model[-1-(layer)].inverse(fwd_output2) #inverse layer outputs
        zz_est_err = np.mean((inv_output.detach().numpy() - fwd_input.detach().numpy())** 2) #reconstruction of z in the middle layers
        print("Block " + str( layer)+ " Propagation" , zyz_err, "Ablation", zz_est_err)
    else:
        print("fwd_output1 is not equal to fwd_output2")

#for layer in range(num_blocks):
#    ## ablation err of isolated block inverse
#    fwd_input = layer_fwd_train[-1-(layer+1)] #forward layer inputs
#    fwd_output1 = layer_fwd_train[-1-(layer)]#forward layer outputs
#    fwd_output2 = INN.model[-1-(layer)].forward(fwd_input)  #forward layer outputs
#    print(str(fwd_output1 == fwd_output2)) # check --> should be same
#    inv_output =  INN.model[-1-(layer)].inverse(fwd_output2) #inverse layer outputs
#    zz_est_err = np.mean((inv_output.detach().numpy() - fwd_input.detach().numpy())** 2) #reconstruction of z in the middle layers
#    print("Block"+str(layer), zz_est_err)
#    # inv_output =  INN.model[-1].inverse(fwd_output) #inverse layer outputs
#


print("Weight matrix inverse:")
for i in range(num_blocks):
    print(torch.linalg.inv(INN.model[i].fc1.weight))
    print(torch.linalg.inv(INN.model[i].fc2.weight))

# print(json.dumps({'train_xyx_err': float(train_xyx_err), 'test_xyx_err': float(test_xyx_err)}))
