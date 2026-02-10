"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from model_utils import injective_pad, ActNorm2D, Split  # .model_utils
from model_utils import squeeze as Squeeze
from model_utils import MaxMinGroup
from spectral_norm_conv_inplace import spectral_norm_conv
from spectral_norm_fc import spectral_norm_fc
# from spectral_norm_mlp_inplace import spectral_norm_mlp
# from spectral_norm_fc import spectral_norm_fc
from matrix_utils import exact_matrix_logarithm_trace, power_series_matrix_logarithm_trace
import pdb
from torch.distributions import constraints


# class LogisticTransform(torch.distributions.Transform):
#     r"""
#     Transform via the mapping :math:`y = \frac{1}{1 + \exp(-x)}` and :math:`x = \text{logit}(y)`.
#     """
#     codomain = constraints.real
#     domain = constraints.unit_interval
#     bijective = True
#     sign = +1
#
#     def __eq__(self, other):
#         return isinstance(other, LogisticTransform)
#
#     def _call(self, x):
#         return x.log() - (-x).log1p()
#
#     def _inverse(self, y):
#         return torch.sigmoid(y)
#
#     def log_abs_det_jacobian(self, x, y):
#         return F.softplus(y) + F.softplus(-y)

def downsample_shape(shape):
    return (shape[0] * 4, shape[1] // 2, shape[2] // 2)


########  i-mlpconv-resnet and i-mlpconv-resnet_block ###########
class Conv2dAsLinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_shape, stride=1, padding=0, bias=True):
        super(Conv2dAsLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Input and output spatial dimensions
        self.input_height, self.input_width = input_shape
        self.output_height = (self.input_height + 2 * padding - self.kernel_size[0]) // stride + 1
        self.output_width = (self.input_width + 2 * padding - self.kernel_size[1]) // stride + 1

        # Total number of input and output neurons
        self.input_neurons = self.in_channels * self.input_height * self.input_width
        self.output_neurons = self.out_channels * self.output_height * self.output_width

        # Define the linear layer
        self.linear = nn.Linear(self.input_neurons, self.output_neurons, bias=bias)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize the convolutional kernel
        conv_kernel = torch.randn(self.out_channels, self.in_channels, *self.kernel_size) * 0.1

        # Initialize the weight matrix
        weight_matrix = torch.zeros(self.output_neurons, self.input_neurons)

        # Map convolution weights to the weight matrix
        for out_ch in range(self.out_channels):
            for in_ch in range(self.in_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        out_index = out_ch * self.output_height * self.output_width + i * self.output_width + j

                        for ki in range(self.kernel_size[0]):
                            for kj in range(self.kernel_size[1]):
                                ii = i * self.stride + ki - self.padding
                                jj = j * self.stride + kj - self.padding

                                if 0 <= ii < self.input_height and 0 <= jj < self.input_width:
                                    in_index = in_ch * self.input_height * self.input_width + ii * self.input_width + jj
                                    weight = conv_kernel[out_ch, in_ch, ki, kj]
                                    weight_matrix[out_index, in_index] += weight

        self.linear.weight.data = weight_matrix
        if self.bias:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        batch_size = x.size(0)
        # print(x.shape)
        
        x = x.view(batch_size, -1)  # Flatten input

        x = self.linear(x)   # Apply linear layer
        # print(x.shape)

        x = x.view(batch_size, self.out_channels, self.output_height, self.output_width)   # Reshape output
        # print(x.shape)
        # print(batch_size, self.out_channels, self.output_height, self.output_width)
        # print("----")
        return x

class mlpconv_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu"):
        super(mlpconv_iresnet_block, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
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

        # Input channels and spatial dimensions
        in_ch, h, w = in_shape

        layers = []
        if input_nonlin:
            layers.append(nonlin())

        # Calculate input shape for Conv2dAsLinear
        input_shape = (h, w)
        # print("stride", stride)

        conv_layer1 = Conv2dAsLinear(in_ch, int_ch, kernel_size=3, input_shape=input_shape, stride=stride, padding=1)
        layers.append(
            self._wrapper_spectral_norm(
                conv_layer1,
            )
        )
        layers.append(nonlin())

        # Calculate new spatial dimensions after first layer
        h1 = (h + 2 * 1 - 3) // stride + 1  # padding=1, kernel_size=3
        w1 = (w + 2 * 1 - 3) // stride + 1

        # Second layer: kernel size 1x1
        input_shape_2 = (h1, w1)
        conv_layer2 = Conv2dAsLinear(int_ch, int_ch, kernel_size=1, input_shape=input_shape_2, stride=1, padding=0)
        layers.append(
            self._wrapper_spectral_norm(
                conv_layer2,
            )
        )
        layers.append(nonlin())

        # Third layer: kernel size 3x3
        conv_layer3 = Conv2dAsLinear(int_ch, in_ch, kernel_size=3, input_shape=input_shape_2, stride=1, padding=1)
        layers.append(
            self._wrapper_spectral_norm(
                conv_layer3,
            )
        )

        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)

        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.).to(x.device)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        
        y = Fx + x
        return y, trace + an_logdet

    def inverse(self, y, maxIter=100):
        x = y
        for _ in range(maxIter):
            summand = self.bottleneck_block(x)
            x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)
        return x

    def _wrapper_spectral_norm(self, layer):
        # Apply spectral normalization to the linear layer inside Conv2dAsLinear
        spectral_norm_fc(layer.linear, self.coeff, n_power_iterations=self.n_power_iter)
        return layer

class mlpconv_iResnet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels, init_ds=1, inj_pad=0,
                 coeff=.9, density_estimation=False, nClasses=None,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 block=mlpconv_iresnet_block,
                 actnorm=True, learn_prior=True,
                 nonlin="relu"):
        super(mlpconv_iResnet, self).__init__()
        assert len(nBlocks) == len(nStrides) == len(nChannels)
        self.init_ds = init_ds
        self.ipad = inj_pad
        self.nBlocks = nBlocks
        self.density_estimation = density_estimation
        self.nClasses = nClasses
        # Parameters for trace estimation
        self.numTraceSamples = numTraceSamples if density_estimation else 0
        self.numSeriesTerms = numSeriesTerms if density_estimation else 0
        self.n_power_iter = n_power_iter

        print(' == Building iResNet %d == ' % (sum(nBlocks) * 3 + 1))
        self.inj_pad = injective_pad(inj_pad)
        in_shape = (in_shape[0] + inj_pad, in_shape[1], in_shape[2])  # Adjust channels

        self.stack, self.in_shapes, self.final_shape = self._make_stack(nChannels, nBlocks, nStrides,
                                                                        in_shape, coeff, block,
                                                                        actnorm, n_power_iter, nonlin)

        # Make prior distribution
        self._make_prior(learn_prior)
        # Make classifier
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
            self.bn1 = nn.BatchNorm2d(final_shape[0], momentum=0.9)
            self.logits = nn.Linear(final_shape[0], nClasses)

    def classifier(self, z):
        out = F.relu(self.bn1(z))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), out.size(1))
        return self.logits(out)

    def prior(self):
        return distributions.Normal(self.prior_mu, torch.exp(self.prior_logstd))

    def _make_stack(self, nChannels, nBlocks, nStrides, in_shape, coeff, block,
                    actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        in_shapes = []
        for i, (int_dim, stride, blocks) in enumerate(zip(nChannels, nStrides, nBlocks)):
            for j in range(blocks):
                in_shapes.append(in_shape)
                block_list.append(block(in_shape, int_dim,
                                        numTraceSamples=self.numTraceSamples,
                                        numSeriesTerms=self.numSeriesTerms,
                                        stride=stride,
                                        input_nonlin=(i + j > 0),  # Add nonlinearity to input for all but first layer
                                        coeff=coeff,
                                        actnorm=actnorm,
                                        n_power_iter=n_power_iter,
                                        nonlin=nonlin))
                # Update in_shape after each block
                h, w = in_shape[1], in_shape[2]
                h = (h + 2 * 1 - 3) // stride + 1  # Assuming padding=1, kernel_size=3
                w = (w + 2 * 1 - 3) // stride + 1
                in_shape = (in_shape[0], h, w)

        return block_list, in_shapes, in_shape

    def get_in_shapes(self):
        return self.in_shapes

    def forward(self, x, ignore_logdet=False):
        if self.ipad != 0:
            x = self.inj_pad.forward(x)

        z = x
        traces = []
        for block in self.stack:
            z, trace = block(z, ignore_logdet=ignore_logdet)
            traces.append(trace)

        # Classification head
        if not self.density_estimation:
            logits = self.classifier(z)
            return logits, z

    def inverse(self, z, max_iter=10):
        with torch.no_grad():
            x = z
            for i in range(len(self.stack)):
                x = self.stack[-1 - i].inverse(x, maxIter=max_iter)

            if self.ipad != 0:
                x = self.inj_pad.inverse(x)
        return x





########  resnet and resnet_block ###########
class conv_resnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu"):
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
        super(conv_resnet_block, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
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

        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride ** 2
        kernel_size1 = 3  # kernel size for first conv
        layers.append(
            nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1))
        layers.append(nonlin())
        kernel_size2 = 1  # kernel size for second conv
        layers.append(nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0))
        layers.append(nonlin())
        kernel_size3 = 3  # kernel size for third conv
        layers.append(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=1))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride == 2:
            x = self.squeeze.forward(x)

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x
        return y, trace + an_logdet

    def inverse(self, y, maxIter=100):  # 100
        # print("start inverse====")
        # inversion of ResNet-block (fixed-point iteration)
        x = y
        # print("maxIter", maxIter)
        for iter_index in range(maxIter):
            summand = self.bottleneck_block(x)
            x = y - summand

        # print("end inverse====")

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        # inversion of squeeze (dimension shuffle)
        if self.stride == 2:
            x = self.squeeze.inverse(x)
        return x

    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff,
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)

class conv_ResNet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels, init_ds=2, inj_pad=0,
                 coeff=.9, density_estimation=False, nClasses=None,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 block = conv_resnet_block,  # todo conv_resnet block layer
                 actnorm=True, learn_prior=True,
                 nonlin="relu"):
        super(conv_ResNet, self).__init__()
        assert len(nBlocks) == len(nStrides) == len(nChannels)
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

        print(' == Building iResNet %d == ' % (sum(nBlocks) * 3 + 1))
        self.init_squeeze = Squeeze(self.init_ds)
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
            self.bn1 = nn.BatchNorm2d(final_shape[0], momentum=0.9)
            self.logits = nn.Linear(final_shape[0], nClasses)  # note here is 10

    def classifier(self, z):
        out = F.relu(self.bn1(z))
        out = F.avg_pool2d(out, out.size(2))
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
        for i, (int_dim, stride, blocks) in enumerate(zip(nChannels, nStrides, nBlocks)):
            for j in range(blocks):
                in_shapes.append(in_shape)
                block_list.append(block(in_shape, int_dim,
                                        numTraceSamples=self.numTraceSamples,
                                        numSeriesTerms=self.numSeriesTerms,
                                        stride=(stride if j == 0 else 1),  # use stride if first layer in block else 1
                                        input_nonlin=(i + j > 0),  # add nonlinearity to input for all but fist layer
                                        coeff=coeff,
                                        actnorm=actnorm,
                                        n_power_iter=n_power_iter,
                                        nonlin=nonlin))
                if stride == 2 and j == 0:
                    in_shape = downsample_shape(in_shape)

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
        if self.init_ds == 2:
            x = self.init_squeeze.forward(x)

        if self.ipad != 0:
            x = self.inj_pad.forward(x)

        z = x
        traces = []
        for block in self.stack:
            z, trace = block(z, ignore_logdet=ignore_logdet)
            traces.append(trace)
        # todo @Muhao：classification vs density estimation
        # no classification head
        # if self.density_estimation:
        #     # add logdets
        #     tmp_trace = torch.zeros_like(traces[0])
        #     for k in range(len(traces)):
        #         tmp_trace += traces[k]
        #
        #     logpz = self.logpz(z)
        #     return z, logpz, tmp_trace

        # classification head
        if not self.density_estimation:
            logits = self.classifier(z)
            return logits, z

    def inverse(self, z, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            x = z
            for i in range(len(self.stack)):
                x = self.stack[-1 - i].inverse(x, maxIter=max_iter)

            if self.ipad != 0:
                x = self.inj_pad.inverse(x)

            if self.init_ds == 2:
                x = self.init_squeeze.inverse(x)
        return x

    # def sample(self, batch_size, max_iter=10):
    #     """sample from prior and invert"""
    #     with torch.no_grad():
    #         # only send batch_size to prior, prior has final_shape as attribute
    #         samples = self.prior().rsample((batch_size,))
    #         samples = samples.view((batch_size,) + self.final_shape)
    #         return self.inverse(samples, max_iter=max_iter)
    #
    # def set_num_terms(self, n_terms):
    #     for block in self.stack:
    #         for layer in block.stack:
    #             layer.numSeriesTerms = n_terms
######## ################### ################### ###########






if __name__ == "__main__":
    # scale = 1.
    # loc = 0.
    # base_distribution = distributions.Uniform(0., 1.)
    # transforms_1 = [distributions.SigmoidTransform().inv, distributions.AffineTransform(loc=loc, scale=scale)]
    # logistic_1 = distributions.TransformedDistribution(base_distribution, transforms_1)

    # transforms_2 = [LogisticTransform(), distributions.AffineTransform(loc=loc, scale=scale)]
    # logistic_2 = distributions.TransformedDistribution(base_distribution, transforms_2)

    # x = torch.zeros(2)
    # print(logistic_1.log_prob(x), logistic_2.log_prob(x))
    # print("x", x.shape)

    diff = lambda x, y: (x - y).abs().sum()
    batch_size = 13
    channels = 3
    h, w = 32, 32
    in_shape = (batch_size, channels, h, w)
    # print("in_shape", in_shape)
    x = torch.randn((batch_size, channels, h, w), requires_grad=True)
    # print("x", x.shape)

    #  section conv_iresnet_block ------------------------------------------------------------
    # block = conv_iresnet_block(in_shape[1:], 32, stride=1, actnorm=True)
    # print(block)
    #
    # out, tr = block(x)#, ignore_logdet=True)
    # print("x", x.shape)
    # print("out", out.shape)
    # print("tr", tr.shape)
    # for i in range(10):
    #     x_re = block.inverse(out, i)
    #     print("x_re", x_re.shape)
    #     print(i, diff(x, x_re))

    # section scale_block------------------------------------------------------------
    # steps = 4
    # int_dim = 32
    # sb = scale_block(steps, in_shape[1:], int_dim, True, 5, 1, True, .9, False, True, True)
    # print(sb)
    #
    # [z1, z2], tr = sb(x)#, ignore_logdet=True)
    # print("x", x.shape)
    # print("z1", z1.shape)
    # print("z2", z2.shape)
    # print("tr", tr.shape)
    #
    # for i in range(10):
    #     x_re = sb.inverse(z1, z2, i)
    #     print(x_re.shape)
    #     print(i, diff(x, x_re))

    # section conv_iResNet------------------------------------------------------------
    resnet = conv_iResNet(in_shape[1:], [4, 4, 4], [1, 2, 2], [32, 32, 32],
                          init_ds=2, density_estimation=False, actnorm=True, nClasses=10)
    # z, lpz, tr = resnet(x)  # , ignore_logdet=True)
    logits, z = resnet(x)
    # print("x", x.shape)
    # print("z", z.shape)
    # print("lpz", lpz.shape)
    # print("tr", tr.shape)

    for i in range(3):
        x_re = resnet.inverse(z, i)
        print("x_re", x_re.shape)
        print("{} iters error {}".format(i, (x - x_re).abs().sum()))
        # exit()
    # section ------------------------------------------------------------
    # resnet = multiscale_conv_iResNet(in_shape[1:], [4, 4, 4], [1, 2, 2], [32, 32, 32],
    #                                  True, 0, .9, True, None, True, 1, 5, True)
    # out, logpz, tr = resnet(x)#, ignore_logdet=True)
    # print(logpz)
    # print(tr)
    # print([o.size() for o in out])
    # print(resnet.z_shapes())
    # sample = resnet.sample(33)
    # print(sample.size())
    # print('multiscale')
    # for i in range(20):
    #     x_re = resnet.inverse(out, i)
    #     print(i, diff(x, x_re))

