#!/usr/bin/env python
# coding: utf-8

# Time independent score based model
# Matt Sampson

# ---------------- #
# import libraries #
# ---------------- #
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from functools import partial
from torchvision.models import ResNet
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm.notebook import tqdm


# ----------------------------------------------------------- #
#                     galaxy zoo dataset                      #
# ----------------------------------------------------------- #
n = 2                # images may be n by 32
im_size = n * 32     # image size
batch_size = 128     # batch size for mini-batch gradient descent (stochastic grad desc.)
transform  = T.Compose([
        ToTensor(),
        T.Resize(size=(im_size, im_size)),
        T.Grayscale(num_output_channels=1),
        T.Normalize((0.5, ), (0.5, ))])
file_name = 'galaxies/'
dataset = datasets.ImageFolder(file_name,
                        transform=transform)
# ----------------------------------------------------------- #
print(dataset)

# ------------------------------------- #
# Split data to training and validation #
# ------------------------------------- #

num_items = len(dataset)
indices = list(range(num_items))
random_state = np.random.get_state()
np.random.seed(2019)
np.random.shuffle(indices)
np.random.set_state(random_state)
train_indices, test_indices = indices[:int(num_items * 0.7)], indices[
            int(num_items * 0.7):int(num_items * 0.8)]
test_dataset = Subset(dataset, test_indices)
train_dataset = Subset(dataset, train_indices)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
testing_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# ----------------------- #
# define the architecture #
# ----------------------- #
"""
This is a re-write of
class's from https://github.com/ermongroup/ncsnv2 with only minor 
alterations. This is based on https://arxiv.org/abs/2006.09011
"""

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out

def conv1x1(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "1x1 convolution"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


def stride_conv3x3(in_planes, out_planes, kernel_size, bias=True, spec_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2,
                     padding=kernel_size // 2, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True, spec_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv

class CRPBlock(nn.Module):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True, spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))
        self.n_stages = n_stages
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU(), spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = normalizer
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))

        self.n_stages = n_stages
        self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)

            x = path + x
        return x


class RCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU(), spec_norm=False):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1), conv3x3(features, features, stride=1, bias=False,
                                                                         spec_norm=spec_norm))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x


class CondRCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU(), spec_norm=False):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
                setattr(self, '{}_{}_conv'.format(i + 1, j + 1),
                        conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

            x += residual
        return x


class MSFBlock(nn.Module):
    def __init__(self, in_planes, features, spec_norm=False):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, spec_norm=False):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm))
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):
    def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True, spec_norm=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                RCUBlock(in_planes[i], 2, 2, act, spec_norm=spec_norm)
            )

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act, spec_norm=spec_norm)

        if not start:
            self.msf = MSFBlock(in_planes, features, spec_norm=spec_norm)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool, spec_norm=spec_norm)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h



class CondRefineBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False, spec_norm=False):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act, spec_norm=spec_norm)
            )

        self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act, spec_norm=spec_norm)

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer, spec_norm=spec_norm)

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act, spec_norm=spec_norm)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False, spec_norm=False):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)
            self.conv = conv
        else:
            conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            if spec_norm:
                conv = spectral_norm(conv)

            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                conv
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                      output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, inputs):
        output = inputs
        output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                      output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return self.conv(output)


class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        if spec_norm:
            self.conv = spectral_norm(self.conv)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, resample=None, act=nn.ELU(),
                 normalization=ConditionalBatchNorm2d, adjust_padding=False, dilation=None, spec_norm=False):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)


    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
                 normalization=nn.BatchNorm2d, adjust_padding=False, dilation=None, spec_norm=False):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation, spec_norm=spec_norm)
            else:
                # conv_shortcut = nn.Conv2d ### Something wierd here.
                conv_shortcut = partial(conv1x1, spec_norm=spec_norm)
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception('invalid resample value')

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)


    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output
    

class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out
    

class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out
    

# ---------------- #
# for images 32**2 #
# ---------------- #
"""
A moderate depth network
which Song+2020 finds works well on
32 by 32 images but not as well on
higher resolutions
"""
    
class NCSNv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_transform = False
        self.rescaled = False
        self.ngf = ngf = 128
        self.num_classes = num_classes = 100
        self.norm = InstanceNorm2dPlus #nn.BatchNorm2d
        self.act = act = nn.ELU()
        image_size = 64
        data_channels = 1
        sigma_begin = 1
        sigma_end = 0.01
        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                               num_classes))).float().to(device)
        self.register_buffer('sigmas', sigmas)

        
        self.begin_conv = nn.Conv2d(data_channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, data_channels, 3, stride=1, padding=1)
        #self.norm = self.normalizer

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        if image_size == 28:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=True, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                              normalization=self.norm, adjust_padding=False, dilation=4),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                              normalization=self.norm, dilation=4)]
            )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output
    
    
# -------------------------- #
# for images 64**2 and above #
# -------------------------- #
"""
A deeper architecture for 64by64 
and above images. Song+2020
uses config files to determine 
some param values, here while in
testing I hard coded some things.

-----------------
Hard coded inputs
-----------------

logit_transform: perform a signmoid like transform
rescaled:        scale the pixel values (I do at read in)
num_classes:     number of elements in sigma/noise vector
norm:            type of normalisation - need to read up more 
on this to see what would be best
act:             activation function, I follow Song+2020 with ELU
data_channels:   all greyscale for us so = 1

"""
    
class NCSNv2Deeper(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_transform = False
        self.rescaled = False
        self.ngf = ngf = 128
        self.num_classes = num_classes = 25
        self.norm = InstanceNorm2dPlus #nn.BatchNorm2d
        self.act = act = nn.ELU()
        image_size = 64
        data_channels = 1
        sigma_begin = 1
        sigma_end = 0.01
        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                               num_classes))).float().to(device)
        self.register_buffer('sigmas', sigmas)

        self.begin_conv = nn.Conv2d(data_channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, data_channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref3 = self.refine3([layer3, ref2], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output


# ------------- #
# loss function #
# ------------- #

def anneal_dsm_score_estimation(scorenet, x , sigmas, labels=None, anneal_power=2.):
    
    if labels is None:
        labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)        # define sigma labels
    used_sigmas = sigmas[labels].view(x.shape[0], *([1] * len(x.shape[1:])))          #
    noise = torch.randn_like(x) * used_sigmas                                         # create noise array
    x_tilde = x + noise                                                               # add noise to image
    target = - 1 / (used_sigmas ** 2) * noise                                         # RHS of eqn 2 term 2
    scores = scorenet(x_tilde, labels)                                                # use NN to calculate score
    target = target.view(target.shape[0], -1)                                         # reshape
    scores = scores.view(scores.shape[0], -1)                                         # reshape
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)



# ----------------------- #
# define training routine # 
# ----------------------- #

# grab the compute we have available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

"""
The parameters used here are especially sensetive

"""


# define training parameters
n_epochs    = 50                              # number of training epochs
lr          = 1e-4                            # learning rate
train       = True                            # train yes/no
store_path  = 'anneal_v2_res64_GPU25.pt'      # model save name
n_steps     = 500000                          # cap total number of steps
sigma_begin = 1                               # start of noise vector        -- dependent on how much variance in p(x)
sigma_end   = 1e-2                            # end of noise vector          -- final amount of noise, maybe we can set even lower?
num_classes = 25                              # num elements in noise vector -- very dependent on image size


# define the model and attach to compute device
score =  NCSNv2Deeper().to(device)
score = torch.nn.DataParallel(score)

# define the training optimiser
optimizer = Adam(score.parameters(), lr=lr)

# define noise vector
sigmas    = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                               num_classes))).float().to(device)

# set progress bar
pbar = tqdm(range(n_epochs), desc='total loss: inf')

# run training if specified
if train:
    best_loss = 1e20
    step = 0
    sigma = 0.01
    for epoch in pbar:
        step += 1
        score.train()
        avg_loss = 0.
        for i, (X, y) in enumerate(loader):
                
            X = X.to(device)
            loss = anneal_dsm_score_estimation(score, X, sigmas, None, 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            if step >= n_steps:
                    break
        
        if (avg_loss < best_loss):
            torch.save(score.state_dict(), store_path)
            pbar.set_description("best model saved - total loss: %d" % avg_loss )
            best_loss = avg_loss
        else:
            pbar.set_description("no save - total loss: %d" % avg_loss )
            

# -------------------------------------------------------------------------------- #
#                              end of training script                              #
# -------------------------------------------------------------------------------- #
