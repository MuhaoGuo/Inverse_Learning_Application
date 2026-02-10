import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.autograd import Variable
# from torch._six import inf
# from models.viz_utils import line_plot, scatter_plot, images_plot
import os
import sys
import math
import numpy as np
import json
import pdb
import time

# from joblib import Parallel, delayed
import multiprocessing

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
        out, z = model(inputs)   # logits, z
        # print("out", out.shape)
        # print("z", z.shape)
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



        # out, z = model(inputs)
        # _, predicted = torch.max(out.data, 1)
        # objective += predicted.eq(targets.data).sum().item()
        # del out, out_bij, _, predicted
        #
        # total += targets.size(0)
        # del inputs, targets

        out, out_bij = model(inputs)
        loss = criterion(out, targets)
        # logging for classification
        _, predicted = torch.max(out.data, 1)
        loss_epoch += loss.item()
        total += targets.size(0)
        #
        # print("predicted", predicted)
        # print("targets", targets)
        # exit()

        correct += predicted.eq(targets.data).cpu().sum()

    # objective = float(objective) / float(total)
    loss_epoch = loss_epoch/len(testloader)
    acc = float(correct) / float(total)

    return loss_epoch, acc

    # print("loss_epoch", loss_epoch)
    # print("acc", acc)
    # exit()
    # return objective
    # # line_plot(viz, "test bits/dim" if args.densityEstimation else "test acc", epoch, objective)
    # print("\n| Validation Epoch #%d\t\t\tobjective =  %.4f" % (epoch, objective), flush=True)
    # if objective > best_result:
    #     print('\n| Saving Best model...\t\t\tobjective = %.4f%%' % (objective), flush=True)
    #     state = {
    #         'model': model if use_cuda else model,
    #         'objective': objective,
    #         'epoch': epoch,
    #     }
    #
    #     try_make_dir(args.save_dir)
    #     torch.save(state, os.path.join(args.save_dir, 'checkpoint.t7'))
    #     best_result = objective
    # else:
    #     print('\n| Not best... {:.4f} < {:.4f}'.format(objective, best_result), flush=True)
    #
    # # log to file
    # test_log.write("{} {}\n".format(epoch, objective))
    # test_log.flush()
    # return best_result


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
