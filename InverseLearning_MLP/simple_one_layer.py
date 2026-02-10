import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
import torch.distributions as distributions
import torch.nn.functional as F
import visdom
import os
import sys
import math
import time
import argparse
import pdb
import torch.nn as nn
import random
import json
import matplotlib.pyplot as plt
from torch.nn import Parameter

import multiprocessing

parser = argparse.ArgumentParser(description='Train i-ResNet/ResNet on Cifar')
parser.add_argument('--save_dir', default="cifar10_100_100", type=str, help='directory to save results') # mnist cifar10
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')  # mnist cifar10

parser.add_argument('--epochs', default=1, type=int, help='number of epochs')  # 200
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')

parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation', action='store_true', help='perform density estimation', default=False)
parser.add_argument('--nBlocks', nargs='+', type=int, default=[1, 0, 0])   # 4, 4, 4   / 1, 0, 0
parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 0, 0])   # 1, 2, 2  / 1, 0, 0
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


def split(x):
    n = int(x.size(1) / 2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()

    def forward(self, x):
        n = int(x.size(1) / 2)
        x1 = x[:, :n, :, :].contiguous()
        x2 = x[:, n:, :, :].contiguous()
        return x1, x2

    def inverse(self, x1, x2):
        return torch.cat((x1, x2), 1)


class squeeze(nn.Module):
    def __init__(self, block_size):
        super(squeeze, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height,
                                                                                                s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


def get_all_params(var, all_params):
    if isinstance(var, Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective, z_list, labels=None):
        raise NotImplementedError

    def reverse_(self, y, objective, labels=None):
        raise NotImplementedError


class ActNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :]

    def shift(self):
        return self._shift[None, :]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum()
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class ActNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm2D, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() * x.size(2) * x.size(3)
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class MaxMinGroup(nn.Module):
    def __init__(self, group_size, axis=-1):
        super(MaxMinGroup, self).__init__()
        self.group_size = group_size
        self.axis = axis

    def forward(self, x):
        maxes = maxout_by_group(x, self.group_size, self.axis)
        mins = minout_by_group(x, self.group_size, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'group_size: {}'.format(self.group_size)


def process_maxmin_groupsize(x, group_size, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % group_size:
        raise ValueError('number of features({}) is not a '
                         'multiple of group_size({})'.format(num_channels, num_channels))
    size[axis] = -1
    if axis == -1:
        size += [group_size]
    else:
        size.insert(axis + 1, group_size)
    return size


def maxout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]

criterion = nn.CrossEntropyLoss()

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def learning_rate(init, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
    return init * math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def out_im(im):
    imc = torch.clamp(im, -.5, .5)
    return imc + .5


def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)


def train(args, model, optimizer, epoch, trainloader, trainset, use_cuda, train_log):
    model.train()
    correct = 0
    total = 0
    loss_epoch = 0

    # update lr for this epoch (for classification only)
    lr = learning_rate(args.lr, epoch)
    update_lr(optimizer, lr)

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        cur_iter = (epoch - 1) * len(trainloader) + batch_idx
        # if first epoch use warmup
        if epoch - 1 <= args.warmup_epochs:
            this_lr = args.lr * float(cur_iter) / (args.warmup_epochs * len(trainloader))
            update_lr(optimizer, this_lr)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()

        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)

        # classification
        print("inputs", inputs.shape)
        out, z = model(inputs)   # logits, z
        print("out", out.shape)
        print("z", z.shape)
        loss = criterion(out, targets)
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        # logging for classification
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 1 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f'
                             % (epoch, args.epochs, batch_idx + 1,
                                (len(trainset) // args.batch) + 1, loss.data.item(),
                                100. * correct.type(torch.FloatTensor) / float(total)))
            sys.stdout.flush()

        loss_epoch += loss.item()

    loss_epoch = loss_epoch/len(trainloader)
    acc = float(correct) / float(total)

    return loss_epoch, acc

def test(best_result, args, model, epoch, testloader, use_cuda, test_log):
    model.eval()
    # objective = 0.
    total = 0
    correct = 0
    loss_epoch = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)

        out, out_bij = model(inputs)
        loss = criterion(out, targets)
        # logging for classification
        _, predicted = torch.max(out.data, 1)
        loss_epoch += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # objective = float(objective) / float(total)
    loss_epoch = loss_epoch/len(testloader)
    acc = float(correct) / float(total)

    return loss_epoch, acc


def _determine_shapes(model):
    in_shapes = model.module.get_in_shapes()
    i = 0
    j = 0
    shape_list = list()
    for key, _ in model.named_parameters():
        if "bottleneck" not in key or "weight" not in key:  # non conv-parameters
            shape_list.append(None)
            continue
        shape_list.append(tuple(in_shapes[j]))
        if i == 2:
            i = 0
            j += 1
        else:
            i += 1
    return shape_list


def _clipping_comp(param, key, coeff, input_shape, use_cuda):
    if "bottleneck" not in key or "weight" not in key:  # non conv-parameters
        return
    # compute SVD via FFT
    convKernel = param.data.cpu().numpy()
    input_shape = input_shape[1:]
    fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
    t_fft_coeff = np.transpose(fft_coeff)
    U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
    if np.max(D) > coeff:
        # first projection onto given norm ball
        Dclip = np.minimum(D, coeff)
        coeffsClipped = np.matmul(U * Dclip[..., None, :], V)
        convKernelClippedfull = np.fft.ifft2(coeffsClipped, axes=[0, 1]).real
        # 1) second projection back to kxk filter kernels
        # and transpose to undo previous transpose operation (used for batch SVD)
        kernelSize1, kernelSize2 = convKernel.shape[2:]
        convKernelClipped = np.transpose(convKernelClippedfull[:kernelSize1, :kernelSize2])
        # reset kernel (using in-place update)
        if use_cuda:
            param.data += torch.tensor(convKernelClipped).float().cuda() - param.data
        else:
            param.data += torch.tensor(convKernelClipped).float() - param.data
    return


def clip_conv_layer(model, coeff, use_cuda):
    shape_list = _determine_shapes(model)
    num_cores = multiprocessing.cpu_count()
    for (key, param), shape in zip(model.named_parameters(), shape_list):
        _clipping_comp(param, key, coeff, shape, use_cuda)
    return


def interpolate(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, out_bij, _ = model(inputs)
        loss = criterion(out, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model

    acc = 100. * correct.type(torch.FloatTensor) / float(total)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.4f%%" % (epoch, loss.data[0], acc), flush=True)

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.4f%%' % (acc), flush=True)
        state = {
            'model': model if use_cuda else model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/' + dataset + os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + fname + '.t7')
        best_acc = acc
    return best_acc


softmax = nn.Softmax(dim=1)

class SpectralNorm(object):

    _version = 1

    def __init__(self, coeff, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.coeff = coeff
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        sigma_log = getattr(module, self.name + '_sigma') # for logging
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log.copy_(sigma.detach())

        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module, name, coeff, n_power_iterations, dim, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(coeff, name, n_power_iterations, dim, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

class SpectralNormLoadStateDictPreHook(object):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[prefix + fn.name + '_v'] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


def spectral_norm_fc(module, coeff, name='weight', n_power_iterations=1, eps=1e-12, dim=None):

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, coeff, n_power_iterations, dim, eps)
    return module



def remove_spectral_norm(module, name='weight'):

    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))

def downsample_shape(shape):
    return (shape[0] * 4, shape[1] // 2, shape[2] // 2)

class mlp_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0, coeff=.97, input_nonlin=True,
                 actnorm=None, n_power_iter=5, nonlin="elu"):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(mlp_iresnet_block, self).__init__()
        #assert stride in (1, 2)
        #self.stride = stride
        #self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
        in_features = in_ch * h * w

        layers = []
        if input_nonlin:
            layers.append(nonlin())

        #in_ch = 32
        layers.append(self._wrapper_spectral_norm(nn.Linear(in_features, in_features), in_features))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """

        Fx = self.bottleneck_block(x)
        y = Fx + x
        return y #trace + an_logdet

    def inverse(self, y, maxIter=100):  # 100
        x = y
        for iter_index in range(maxIter):
            summand = self.bottleneck_block(x)
            x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        return x

    def _wrapper_spectral_norm(self, layer, in_features):
        return spectral_norm_fc(layer, self.coeff, n_power_iterations=self.n_power_iter)

class mlp_iResNet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels, init_ds=2, inj_pad=0,
                 coeff=.9, density_estimation=False, nClasses=None,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 block=mlp_iresnet_block,  # todo conv_iresnet block layer
                 actnorm=None, learn_prior=True,
                 nonlin="relu"):
        super(mlp_iResNet, self).__init__()
        #assert len(nBlocks) == len(nStrides) == len(nChannels)
        assert init_ds in (1, 2), "can only squeeze by 2"
        self.init_ds = init_ds
        self.ipad = inj_pad
        self.nBlocks = nBlocks
        self.density_estimation = density_estimation
        self.nClasses = nClasses
        # parameters for trace estimation
        self.numTraceSamples = numTraceSamples if density_estimation else 0
        self.numSeriesTerms = numSeriesTerms if density_estimation else 0
        self.n_power_iter = n_power_iter

        print('')
        print(' == Building iResNet %d == ' % (sum(nBlocks) * 3 + 1))
        #self.init_squeeze = Squeeze(self.init_ds)
        self.inj_pad = injective_pad(inj_pad)
        if self.init_ds == 2:
            in_shape = downsample_shape(in_shape)
        in_shape = (in_shape[0] + inj_pad, in_shape[1], in_shape[2])  # adjust channels

        self.stack, self.in_shapes, self.final_shape = self._make_stack(nChannels, nBlocks, nStrides,
                                                                        in_shape, coeff, block,
                                                                        actnorm, n_power_iter, nonlin)

        # make prior distribution
        self._make_prior(learn_prior)
        # make classifier
        self._make_classifier(self.final_shape, nClasses)
        assert (nClasses is not None or density_estimation), "Must be either classifier or density estimator"

    def _make_prior(self, learn_prior):
        dim = np.prod(self.in_shapes[0])
        self.prior_mu = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)
        self.prior_logstd = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)

    def _make_classifier(self, final_shape, nClasses):
        if nClasses is None:
            self.logits = None
        else:
            print(final_shape)
            self.bn1 = nn.BatchNorm1d(final_shape[0]*final_shape[1]*final_shape[2], momentum=0.9)
            self.logits = nn.Linear(final_shape[0]*final_shape[1]*final_shape[2], nClasses)  # note here is 10

    def classifier(self, z):
        out = F.relu(self.bn1(z))
        #out = F.avg_pool1d(out, out.size(2))
        out = out.view(out.size(0), out.size(1))
        return self.logits(out)

    def prior(self):
        return distributions.Normal(self.prior_mu, torch.exp(self.prior_logstd))

    # def logpz(self, z):
    #     return self.prior().log_prob(z.view(z.size(0), -1)).sum(dim=1)

    def _make_stack(self, nChannels, nBlocks, nStrides, in_shape, coeff, block,
                    actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        in_shapes = []
        for i, (int_dim, blocks) in enumerate(zip(nChannels, nBlocks)):
            for j in range(blocks):
                in_shapes.append(in_shape)
                block_list.append(block(in_shape, int_dim,
                                        numTraceSamples=self.numTraceSamples,
                                        numSeriesTerms=self.numSeriesTerms,  # use stride if first layer in block else 1
                                        input_nonlin=(i + j > 0),  # add nonlinearity to input for all but fist layer
                                        coeff=coeff,
                                        actnorm=actnorm,
                                        n_power_iter=n_power_iter,
                                        nonlin=nonlin))
                #if stride == 2 and j == 0:
                    #in_shape = downsample_shape(in_shape)

        return block_list, in_shapes, in_shape

    def get_in_shapes(self):
        return self.in_shapes

    # def inspect_singular_values(self):
    #     i = 0
    #     j = 0
    #     params = [v for v in self.state_dict().keys()
    #               if "bottleneck" in v and "weight_orig" in v
    #               and not "weight_u" in v
    #               and not "bn1" in v
    #               and not "linear" in v]
    #     print(len(params))
    #     print(len(self.in_shapes))
    #     svs = []
    #     for param in params:
    #         input_shape = tuple(self.in_shapes[j])
    #         # get unscaled parameters from state dict
    #         convKernel_unscaled = self.state_dict()[param].cpu().numpy()
    #         # get scaling by spectral norm
    #         sigma = self.state_dict()[param[:-5] + '_sigma'].cpu().numpy()
    #         convKernel = convKernel_unscaled / sigma
    #         # compute singular values
    #         input_shape = input_shape[1:]
    #         fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
    #         t_fft_coeff = np.transpose(fft_coeff)
    #         D = np.linalg.svd(t_fft_coeff, compute_uv=False, full_matrices=False)
    #         Dflat = np.sort(D.flatten())[::-1]
    #         print("Layer " + str(j) + " Singular Value " + str(Dflat[0]))
    #         svs.append(Dflat[0])
    #         if i == 2:
    #             i = 0
    #             j += 1
    #         else:
    #             i += 1
    #     return svs

    def forward(self, x, ignore_logdet=False):
        """ iresnet forward """
        #if self.init_ds == 2:
            #x = self.init_squeeze.forward(x)

        #if self.ipad != 0:
            #x = self.inj_pad.forward(x)
        x = x.view(x.size(0), -1)
        print(x.size)

        z = x
        traces = []
        for block in self.stack:
            z = block(z, ignore_logdet=ignore_logdet)
        if not self.density_estimation:
            logits = self.classifier(z)
            return logits, z

    def inverse(self, z, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            x = z
            for i in range(len(self.stack)):
                x = self.stack[-1 - i].inverse(x, maxIter=max_iter)
        return x

    
print("task", "classification")
clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
transform_train = transforms.Compose(train_chain + clf_chain)
transform_test = transforms.Compose(test_chain + clf_chain)
    
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

model = mlp_iResNet(nBlocks=args.nBlocks,
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
print(model)

if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
    f.write(json.dumps(args.__dict__))

train_log = open(os.path.join(args.save_dir, "train_log.txt"), 'w')
elapsed_time = 0
test_objective = -np.inf

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

# init actnrom parameters
init_batch = get_init_batch(trainloader, args.init_batch)
print("initializing actnorm parameters...")
with torch.no_grad():
    model(init_batch, ignore_logdet=True)
print("initialized")

use_cuda = torch.cuda.is_available()
in_shapes = model.get_in_shapes()
optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


#section training ############################
record = {}
record["train_epoch_loss"] = []
record["train_epoch_acc"] = []
record["train_epoch_time"] = []
for epoch in range(1, 1 + args.epochs):
    start_time = time.time()
    loss_epoch, acc = train(args, model, optimizer, epoch, trainloader, trainset, use_cuda, train_log)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
    record["train_epoch_loss"].append(loss_epoch)
    record["train_epoch_acc"].append(acc)
    record["train_epoch_time"].append(epoch_time)

# Save record
json_string = json.dumps(record)
with open(args.save_dir + '/record.json', 'w') as file:
    file.write(json_string)

# Save the model state
torch.save(model.state_dict(), args.save_dir + '/model_state_dict.pth')
# load model  -----------------------------------
state_dict = torch.load(args.save_dir + '/model_state_dict.pth')
model.load_state_dict(state_dict)


#########################################################
# section testing #####################################
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

# section inverse part ###################################
print("------inverse part---------------")
inverse_iters = 1
record_inverse = {}
record_inverse["inverse_train_loss"] = []
record_inverse["inverse_train_time"] = []
record_inverse["inverse_test_loss"] = []
record_inverse["inverse_test_time"] = []
record_inverse["inverse_iters"] = inverse_iters
criterion_inverse = nn.MSELoss()

##section inverse train data ###################################
inverse_train_loss = 0
start_time = time.time()
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
    #inputs2 = inputs.view(inputs.size(0), -1)
    #inputs_expanded = inputs2.unsqueeze(0)
    inputs_repeated = inputs.repeat(12, 1, 1, 1)
    out, z1 = model(inputs)  # logits, z
    x_re = model.inverse(z1, inverse_iters)
    x_re_reshaped = x_re.view(-1, 3, 32, 32)
    mse_inverse = criterion_inverse(inputs_repeated, x_re_reshaped).item()
    inverse_train_loss += mse_inverse

    if batch_idx == 1:  ########## visualization of train images ###############
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
        x_re2 = x_re.view(-1, 3, 32, 32)
        for i in range(1, 11):
            for j in range(1, 11):
                image = x_re2[i * j - 1, :, :, :].permute(1, 2, 0).detach().numpy().astype(np.float64)
                axs[i - 1][j - 1].imshow(image)
                axs[i - 1][j - 1].set_xticklabels([])
                axs[i - 1][j - 1].set_yticklabels([])
        plt.savefig(args.save_dir + '/train_reconstruct.png', format='png')

inverse_train_loss = inverse_train_loss / len(trainloader)
inverse_time = time.time() - start_time
record_inverse["inverse_train_loss"].append(inverse_train_loss)
record_inverse["inverse_train_time"].append(inverse_time)

## section inverse test data ###################################
inverse_test_loss = 0
start_time = time.time()
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
    inputs_repeated = inputs.repeat(12, 1, 1, 1)
    out, z1 = model(inputs)  # logits, z
    x_re = model.inverse(z1, inverse_iters)
    x_re_reshaped = x_re.view(-1, 3, 32, 32)
    mse_inverse = criterion_inverse(inputs_repeated, x_re_reshaped).item()
    inverse_test_loss += mse_inverse

    if batch_idx == 1: ########## visualization of test images ###############
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
        print(x_re)
        x_re2 = x_re.view(-1, 3, 32, 32)
        for i in range(1, 11):
            for j in range(1, 11):
                image = x_re2[i * j - 1, :, :, :].permute(1, 2, 0).detach().numpy().astype(np.float64)
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
