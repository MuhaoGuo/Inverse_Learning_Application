import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
import visdom
import os
import sys
import time
import argparse
import pdb
import torch.nn as nn
import random
import json
from utils_cifar import train, test, std, mean, get_hms, interpolate
from conv_iResNet import conv_iResNet as iResNet
# from models.conv_iResNet import multiscale_conv_iResNet as multiscale_iResNet
import matplotlib.pyplot as plt


'''
只load model， 做 test使用， 但貌似不work，不知道为什么。
'''


parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--save_dir', default="cifar10_200", type=str, help='directory to save results') # cifar10_100_444
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')  # mnist cifar10

parser.add_argument('--epochs', default=100, type=int, help='number of epochs')  # 200
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation', action='store_true', help='perform density estimation', default=False)
parser.add_argument('--nBlocks', nargs='+', type=int, default=[1, 0, 0])   # 4, 4, 4
parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 0, 0])   # 1, 2, 2  只有最简单的 cifar10_200 是1, 0, 0
parser.add_argument('--nChannels', nargs='+', type=int, default=[16, 64, 128])  #16, 64, 256
parser.add_argument('--coeff', default=0.9, type=float, help='contraction coefficient for linear layers')
parser.add_argument('--numTraceSamples', default=1, type=int, help='number of samples used for trace estimation')
parser.add_argument('--numSeriesTerms', default=1, type=int, help='number of terms used in power series for matrix log')
parser.add_argument('--powerIterSpectralNorm', default=5, type=int, help='number of power iterations used for spectral norm')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='coefficient for weight decay')
parser.add_argument('--init_batch', default=1024, type=int, help='init batch size')
parser.add_argument('--init_ds', default=2, type=int, help='initial downsampling')
parser.add_argument('--inj_pad', default=0, type=int, help='initial inj padding')
parser.add_argument('--warmup_epochs', default=10, type=int, help='epochs for warmup')

args = parser.parse_args()
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.backends.cudnn.deterministic = True

train_chain = [transforms.Pad(4, padding_mode="symmetric"),
               transforms.RandomCrop(32),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor()]
test_chain = [transforms.ToTensor()]

print("task", "classification")
clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
transform_train = transforms.Compose(train_chain + clf_chain)
transform_test = transforms.Compose(test_chain + clf_chain)


# if dataset == 'mnist':
#     # assert args.densityEstimation, "Currently mnist is only supported for density estimation"
#     mnist_transforms = [transforms.Pad(2, 0), transforms.ToTensor(), lambda x: x.repeat((3, 1, 1))]
#     transform_train_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
#     transform_test_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
#     trainset = torchvision.datasets.MNIST(
#         root='./data', train=True, download=True, transform=transform_train_mnist)
#     testset = torchvision.datasets.MNIST(
#         root='./data', train=False, download=False, transform=transform_test_mnist)
#     nClasses = 10
#     in_shape = (1, 28, 28)  # NOTE (3, 32, 32) --> (1, 28, 28)

if args.dataset == 'cifar10':
    print("dataset", "cifar10")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    nClasses = 10
    in_shape = (3, 32, 32)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                          shuffle=True, num_workers=0,
                                          worker_init_fn=np.random.seed(1234))  # num_workers=0
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                         shuffle=False, num_workers=0,
                                         worker_init_fn=np.random.seed(1234))  # num_workers=0


noActnorm = False
fixedPrior = False
nonlin = 'elu'
model = iResNet(nBlocks=args.nBlocks,
                nStrides=args.nStrides,
                nChannels=args.nChannels,
                nClasses=nClasses,
                init_ds=args.init_ds,
                inj_pad=args.inj_pad,
                in_shape=in_shape,
                coeff=args.coeff,
                numTraceSamples=args.numTraceSamples,
                numSeriesTerms=args.numSeriesTerms,
                n_power_iter=args.powerIterSpectralNorm,
                density_estimation=args.densityEstimation,
                actnorm=(not noActnorm),
                learn_prior=(not fixedPrior),
                nonlin=nonlin)

# print(model)
use_cuda = torch.cuda.is_available()


def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init
    """
    batches = []
    seen = 0
    for x, y in dataloader:
        batches.append(x)
        seen += x.size(0)
        if seen >= batch_size:
            break
    batch = torch.cat(batches)
    return batch



# load model  -----------------------------------
state_dict = torch.load(args.save_dir + '/model_state_dict.pth')
model.load_state_dict(state_dict)

## option2 load model  -----------------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load(args.save_dir + '/checkpoint.t7', map_location=device)
# print(checkpoint["model"])
# model = checkpoint["model"]

testloader = trainloader

# init actnrom parameters
init_batch = get_init_batch(testloader, args.init_batch)
print("initializing actnorm parameters...")
with torch.no_grad():
    model(init_batch, ignore_logdet=True)
print("initialized")


print('Testing model--------------------------')
start_time = time.time()
test_objective = -np.inf
test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
loss_epoch, acc = test(test_objective, args, model, args.epochs, testloader, use_cuda, test_log)

print("test loss_epoch", loss_epoch)
print("test acc", acc)
# print('* Test results : objective = %.2f%%' % (test_objective))
with open(os.path.join(args.save_dir, 'final.txt'), 'w') as f:
    f.write(str(loss_epoch) + "\n" + str(acc))
test_time = time.time() - start_time
print("test_time", test_time)

print("------inverse part---------------")
inverse_iters = 1
record_inverse = {}
record_inverse["inverse_train_loss"] = []
record_inverse["inverse_train_time"] = []
record_inverse["inverse_test_loss"] = []
record_inverse["inverse_test_time"] = []
record_inverse["inverse_iters"] = inverse_iters
criterion_inverse = nn.MSELoss()

## section train data
inverse_train_loss = 0
start_time = time.time()
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
    out, z = model(inputs)  # logits, z
    x_re = model.inverse(z, inverse_iters)


    print("inputs", inputs.shape)
    print("targets", targets.shape)
    print("out", out.shape)
    print("z", z.shape)
    print("x_re", x_re.shape)

    print("inputs", inputs[3].detach().numpy())
    print("targets", targets[3].detach().numpy())
    print("out", out[3].detach().numpy())
    print("z", z[3].detach().numpy())
    print("x_re", x_re[3].detach().numpy())
    exit()

    mse_inverse = criterion_inverse(inputs, x_re).item()
    inverse_train_loss += mse_inverse

    if batch_idx == 0:  ########## visualization of train images ###############
        fig, axs = plt.subplots(10, 10, figsize=(10,10))
        fig.suptitle('train_inputs')
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1, 11):
            for j in range(1, 11):
                image = inputs[i * j - 1, :, :, :].permute(1, 2, 0).detach().numpy().astype(np.float64)
                axs[i - 1][j - 1].imshow(image)
                axs[i - 1][j - 1].set_xticklabels([])
                axs[i - 1][j - 1].set_yticklabels([])
        plt.savefig(args.save_dir + '/train_inputs.png', format='png')

        fig, axs = plt.subplots(10, 10, figsize=(10,10))
        fig.suptitle('train_reconstruct')
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1, 11):
            for j in range(1, 11):
                image = x_re[i * j - 1, :, :, :].permute(1, 2, 0).detach().numpy().astype(np.float64)
                axs[i - 1][j - 1].imshow(image)
                axs[i - 1][j - 1].set_xticklabels([])
                axs[i - 1][j - 1].set_yticklabels([])
        plt.savefig(args.save_dir + '/train_reconstruct.png', format='png')

inverse_train_loss = inverse_train_loss / len(trainloader)
inverse_time = time.time() - start_time
record_inverse["inverse_train_loss"].append(inverse_train_loss)
record_inverse["inverse_train_time"].append(inverse_time)

## section test data
inverse_test_loss = 0
start_time = time.time()
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
    out, z = model(inputs)  # logits, z
    x_re = model.inverse(z, inverse_iters)
    mse_inverse = criterion_inverse(inputs, x_re).item()
    inverse_test_loss += mse_inverse

    if batch_idx == 0: ########## visualization of test images ###############
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        fig.suptitle('test_inputs')
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1, 11):
            for j in range(1, 11):
                image = inputs[i * j - 1, :, :, :].permute(1, 2, 0).detach().numpy().astype(np.float64)
                axs[i - 1][j - 1].imshow(image)
                axs[i - 1][j - 1].set_xticklabels([])
                axs[i - 1][j - 1].set_yticklabels([])
        plt.savefig(args.save_dir + '/test_inputs.png', format='png')

        fig, axs = plt.subplots(10, 10, figsize=(10,10))
        fig.suptitle('test_reconstruct')
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(1, 11):
            for j in range(1, 11):
                image = x_re[i * j - 1, :, :, :].permute(1, 2, 0).detach().numpy().astype(np.float64)
                axs[i - 1][j - 1].imshow(image)
                axs[i - 1][j - 1].set_xticklabels([])
                axs[i - 1][j - 1].set_yticklabels([])
        plt.savefig(args.save_dir + '/test_reconstruct.png', format='png')

inverse_test_loss = inverse_test_loss / len(testloader)
inverse_time = time.time() - start_time
record_inverse["inverse_test_loss"].append(inverse_test_loss)
record_inverse["inverse_test_time"].append(inverse_time)

# Save record_inverse
json_string = json.dumps(record_inverse)
with open(args.save_dir + '/record_inverse.json', 'w') as file:
    file.write(json_string)


