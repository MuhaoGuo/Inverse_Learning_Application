import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
from model_utils import injective_pad, ActNorm2D, Split  # .model_utils
from model_utils import squeeze as Squeeze
from model_utils import MaxMinGroup
from spectral_norm_mlp_inplace import spectral_norm_mlp
from spectral_norm_fc import spectral_norm_fc
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

class mlp_DipDNN_block(nn.Module):
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
        super(mlp_DipDNN_block, self).__init__()
        #assert stride in (1, 2)
        #self.stride = stride
        #self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "LeakyReLU": nn.LeakyReLU,
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
            # layers.append(nonlin(negative_slope=0.01))
            layers.append(nonlin(negative_slope=0.01))

  
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.fc2 = nn.Linear(in_features, in_features, bias=True)
        self.register_buffer('mask_tril', torch.tril(torch.ones_like(self.fc1.weight)))
        self.register_buffer('mask_triu', torch.triu(torch.ones_like(self.fc2.weight)))
        self.fc1.weight = nn.Parameter(torch.mul(self.fc1.weight, self.mask_tril))
        self.fc2.weight = nn.Parameter(torch.mul(self.fc2.weight, self.mask_triu))

        layers.append(self.fc1)
        layers.append(nonlin(negative_slope=0.01))  # negative_slope=0.01
        layers.append(self.fc2)
        layers.append(nonlin(negative_slope=0.01))  # negative_slope=0.01
        self.bottleneck_block = nn.Sequential(*layers)


        # #in_ch = 32
        # #kernel_size1 = 3  # kernel size for first conv
        # # layers.append(self._wrapper_spectral_norm(nn.Linear(in_features, in_features), in_features))
        # #layers.append(nonlin())
        # #kernel_size2 = 1  # kernel size for second conv
        # layers.append(self._wrapper_spectral_norm(nn.Linear(in_features, in_features), in_features))
        # layers.append(nonlin())
        # #kernel_size3 = 3  # kernel size for third conv
        # layers.append(self._wrapper_spectral_norm(nn.Linear(in_features, in_features), in_features))
        # layers.append(nonlin())
        # #kernel_size3 = 3  # kernel size for third conv
        # layers.append(self._wrapper_spectral_norm(nn.Linear(in_features, in_features), in_features))
        # self.bottleneck_block = nn.Sequential(*layers)

        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        #if self.stride == 2:
            #x = self.squeeze.forward(x)


        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)
        # Compute approximate trace for use in training
        #if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            #trace = torch.tensor(0.)
        #else:
            #trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx #+ x
        return y #trace + an_logdet

    def inverse(self, y, maxIter=100):  # 100
        fc2_W_T = torch.linalg.inv(torch.mul(self.fc2.weight, self.mask_triu))
        fc2_inv = F.linear(F.leaky_relu(y, negative_slope=1/0.01) - self.fc2.bias, fc2_W_T)
        fc1_W_T = torch.linalg.inv(torch.mul(self.fc1.weight, self.mask_tril))
        fc1_inv = F.linear(F.leaky_relu(fc2_inv, negative_slope=1/0.01) - self.fc1.bias, fc1_W_T)
        x = fc1_inv

        # x = y
        # for iter_index in range(maxIter):
        #     summand = self.bottleneck_block(x)
        #     x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        # inversion of squeeze (dimension shuffle)
        #if self.stride == 2:
            #x = self.squeeze.inverse(x)
        return x

    def _wrapper_spectral_norm(self, layer, in_features):
        #if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
        return spectral_norm_fc(layer, self.coeff, n_power_iterations=self.n_power_iter)
        #else:
            # use spectral norm based on conv, because bound not tight
            #return spectral_norm_conv(layer, self.coeff, shapes,
                                      #n_power_iterations=self.n_power_iter)

class mlp_DipDNN(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels, init_ds=2, inj_pad=0,
                 coeff=.9, density_estimation=False, nClasses=None,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 block=mlp_DipDNN_block,  # todo conv_iresnet block layer
                 actnorm=None, learn_prior=True,
                 nonlin="relu"):
        super(mlp_DipDNN, self).__init__()
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
        self.dropout = nn.Dropout(p=0.2)
        
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
            self.feature_reduction = nn.Linear(final_shape[0]*final_shape[1]*final_shape[2], 64)
            self.logits = nn.Linear(64, nClasses)  # note here is 10

    def classifier(self, z):
        # # out = F.relu(self.bn1(z))
        # out = F.leaky_relu(self.bn1(z),negative_slope=0.01)
        # #out = F.avg_pool1d(out, out.size(2))
        # out = out.view(out.size(0), out.size(1))
        # out = z
        out = self.feature_reduction(z)
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

        z = x
        traces = []
        for block in self.stack:
            z = block(z, ignore_logdet=ignore_logdet)
            #traces.append(trace)
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
            z = self.dropout(z)
            logits = self.classifier(z)
            return logits, z

    def inverse(self, z, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            x = z
            for i in range(len(self.stack)):
                x = self.stack[-1 - i].inverse(x, maxIter=max_iter)

            #if self.ipad != 0:
                #x = self.inj_pad.inverse(x)

            #if self.init_ds == 2:
                #x = self.init_squeeze.inverse(x)
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


