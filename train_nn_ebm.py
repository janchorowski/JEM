# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import numpy as np
import wideresnet
import json
# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
from sklearn import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vbnorm import VirtualBatchNormNN
from losses import VATLoss, LDSLoss, sliced_score_matching_vr, \
    sliced_score_matching, denoising_score_matching
import regression_datasets
from matplotlib.colors import ListedColormap


import toy_data
TOY_DSETS = ("moons", "circles", "8gaussians", "pinwheel", "2spirals", "checkerboard", "rings", "swissroll")
REG_DSETS = {"concrete": 8, "protein": 9, "navy": 16, "power_plant": 4, "year": 90}

class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class Swish(nn.Module):
    def __init__(self, dim=-1):
        super(Swish, self).__init__()
        if dim > 0:
            self.beta = nn.Parameter(t.ones((dim,)))
        else:
            self.beta = t.ones((1,))
    def forward(self, x):
        if len(x.size()) == 2:
            return x * t.sigmoid(self.beta[None, :] * x)
        else:
            return x * t.sigmoid(self.beta[None, :, None, None] * x)

class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return t.sin(x)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, extra_layers, use_vbnorm=False, ref_x=None, n_channels_in=1):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.use_vbnorm = use_vbnorm

        self.n_channels_in = n_channels_in

        affine = True
        if args.no_param_bn:
            affine = False

        layer_in = nn.Linear(input_size * n_channels_in, hidden_size)
        if args.sine:
            unif_bound = np.sqrt(6.0 / input_size)
            first_layer_scalar = 30
            nn.init.uniform_(layer_in.weight, -unif_bound, unif_bound) * first_layer_scalar
        self.layers.append(layer_in)
        self.ref_x = ref_x

        if use_vbnorm:
            assert ref_x is not None
            self.layers.append(VirtualBatchNormNN(hidden_size))
        elif args.batch_norm:
            self.layers.append(nn.BatchNorm1d(num_features=hidden_size, affine=affine))

        if args.swish:
            self.layers.append(Swish(hidden_size))
        elif args.sine:
            self.layers.append(Sine())
        elif args.softplus:
            self.layers.append(nn.Softplus())
        elif args.leaky_relu:
            self.layers.append(nn.LeakyReLU())
        else:
            self.layers.append(nn.ReLU())

        if args.dropout:
            self.layers.append(nn.Dropout(p=0.5))

        for i in range(extra_layers):
            layer = nn.Linear(hidden_size, hidden_size)
            if args.sine:
                nn.init.uniform_(layer.weight, -unif_bound, unif_bound)
            self.layers.append(layer)
            # self.layers.append(nn.Linear(hidden_size, hidden_size))
            if not args.first_layer_bn_only:
                if use_vbnorm:
                    self.layers.append(VirtualBatchNormNN(hidden_size))
                elif args.batch_norm:
                    self.layers.append(nn.BatchNorm1d(num_features=hidden_size, affine=affine))
            if args.swish:
                self.layers.append(Swish(hidden_size))
            elif args.sine:
                self.layers.append(Sine())
            elif args.softplus:
                self.layers.append(nn.Softplus())
            elif args.leaky_relu:
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers.append(nn.ReLU())
            if args.dropout:
                self.layers.append(nn.Dropout(p=0.5))

        # Note output layer not needed here because it is done in class F


    def forward(self, x, y=None):
        if args.vbnorm:
            ref_x = self.ref_x
            if len(ref_x.shape) > 2:
                if self.n_channels_in > 1:
                    ref_x = ref_x.reshape(-1, ref_x.shape[-1] ** 2 * self.n_channels_in)
                else:
                    ref_x = ref_x.reshape(-1, ref_x.shape[-1]**2)
        if len(x.shape) > 2:
            if self.n_channels_in > 1:
                x = x.reshape(-1, x.shape[-1]**2 * self.n_channels_in)
            else:
                x = x.reshape(-1, x.shape[-1]**2)
        for layer in self.layers:
            if isinstance(layer, VirtualBatchNormNN):
                assert ref_x is not None
                ref_x, mean, mean_sq = layer(ref_x, None, None)
                x, _, _ = layer(x, mean, mean_sq)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else: # now includes ReLU/activation functions
                if args.vbnorm:
                    ref_x = layer(ref_x)
                x = layer(x)
        output = x
        return output


def conv_lrelu_bn_block(channels, n_units, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(channels, n_units, kernel, padding=padding),
        nn.LeakyReLU(negative_slope=0.1),
        nn.BatchNorm2d(num_features=n_units)
    )

def conv_lrelu_block(channels, n_units, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(channels, n_units, kernel, padding=padding),
        nn.LeakyReLU(negative_slope=0.1)
    )

def conv_swish_block(channels, n_units, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(channels, n_units, kernel, padding=padding),
        Swish(n_units)
    )

class ConvLarge(nn.Module):
    # Based on VAT paper, what they call "ConvLarge"
    def __init__(self, avg_pool_kernel=6):
        super(ConvLarge, self).__init__()
        self.layers = nn.ModuleList()

        if args.swish:
            self.layers.append(conv_swish_block(args.n_ch, n_units=128, kernel=3,
                                 padding=1))
            self.layers.append(conv_swish_block(128, 128, kernel=3, padding=1))
            self.layers.append(conv_swish_block(128, 128, kernel=3, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if not args.cnn_no_dropout:
                self.layers.append(nn.Dropout2d(p=0.5))
            self.layers.append(conv_swish_block(128, 256, kernel=3, padding=1))
            self.layers.append(conv_swish_block(256, 256, kernel=3, padding=1))
            self.layers.append(conv_swish_block(256, 256, kernel=3, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if not args.cnn_no_dropout:
                self.layers.append(nn.Dropout2d(p=0.5))
            self.layers.append(
                conv_swish_block(256, 512, kernel=3, padding=0))
            self.layers.append(
                conv_swish_block(512, 256, kernel=1, padding=0))
            self.layers.append(
                conv_swish_block(256, 128, kernel=1, padding=0))
        else:
            if args.cnn_no_bn:
                self.layers.append(conv_lrelu_block(args.n_ch, n_units=128, kernel=3,
                                        padding=1))
                self.layers.append(conv_lrelu_block(128, 128, kernel=3, padding=1))
                self.layers.append(conv_lrelu_block(128, 128, kernel=3, padding=1))
            else:
                self.layers.append(conv_lrelu_bn_block(args.n_ch, n_units=128, kernel=3,
                                    padding=1))
                self.layers.append(conv_lrelu_bn_block(128, 128, kernel=3, padding=1))
                self.layers.append(conv_lrelu_bn_block(128, 128, kernel=3, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if not args.cnn_no_dropout:
                self.layers.append(nn.Dropout2d(p=0.5))
            if args.cnn_no_bn:
                self.layers.append(conv_lrelu_block(128, 256, kernel=3, padding=1))
                self.layers.append(conv_lrelu_block(256, 256, kernel=3, padding=1))
                self.layers.append(conv_lrelu_block(256, 256, kernel=3, padding=1))
            else:
                self.layers.append(conv_lrelu_bn_block(128, 256, kernel=3, padding=1))
                self.layers.append(conv_lrelu_bn_block(256, 256, kernel=3, padding=1))
                self.layers.append(conv_lrelu_bn_block(256, 256, kernel=3, padding=1))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if not args.cnn_no_dropout:
                self.layers.append(nn.Dropout2d(p=0.5))
            if args.cnn_no_bn:
                self.layers.append(
                    conv_lrelu_block(256, 512, kernel=3, padding=0))
                self.layers.append(
                    conv_lrelu_block(512, 256, kernel=1, padding=0))
                self.layers.append(
                    conv_lrelu_block(256, 128, kernel=1, padding=0))
            else:
                self.layers.append(
                    conv_lrelu_bn_block(256, 512, kernel=3, padding=0))
                self.layers.append(
                    conv_lrelu_bn_block(512, 256, kernel=1, padding=0))
                self.layers.append(
                    conv_lrelu_bn_block(256, 128, kernel=1, padding=0))

        self.layers.append(nn.AvgPool2d(kernel_size=avg_pool_kernel))
        # nn.Linear(128, 10) No final linear, done in class F

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        out = out.squeeze()
        return out


class SmallConvNet(nn.Module):
    def __init__(self, sm_dim, spectral_norm=False):
        super(SmallConvNet, self).__init__()
        C1S, C2S, F1S = 20, 50, sm_dim
        if spectral_norm:
            N = nn.utils.spectral_norm
        else:
            N = lambda x: x
        self.conv1 = N(nn.Conv2d(3, C1S, 5, 1))
        self.conv2 = N(nn.Conv2d(C1S, C2S, 5, 1))
        last_dim = 5*5*C2S
        layers = []
        while last_dim // 4 > sm_dim:
            new_dim = last_dim //4
            layers.append(N(nn.Linear(last_dim, new_dim)))
            layers.append(nn.ReLU())
            last_dim = new_dim
        layers.append(N(nn.Linear(last_dim, sm_dim)))
        self.fc1 = nn.Sequential(*layers)
        

    def forward(self, x):
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x = tnnF.relu(self.conv1(x))
        x = tnnF.max_pool2d(x, 2, 2)
        x = tnnF.relu(self.conv2(x))
        x = tnnF.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = self.fc1(x)
        return x


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, im_sz=32, use_nn=False, input_size=None, n_classes=10, ref_x=None, use_cnn=False):
        if input_size is not None:
            assert use_nn == True #input size is for non-images, ie non-conv.
        super(F, self).__init__()
        # print(input_size)
        if use_cnn:
            # print("Using ConvLarge")
            # self.f = ConvLarge(avg_pool_kernel=args.cnn_avg_pool_kernel)
            self.f = SmallConvNet(args.sm_dim)
            self.f.last_dim = args.sm_dim
        elif use_nn:
            hidden_units = args.nn_hidden_size

            use_vbnorm = False
            if args.vbnorm:
                use_vbnorm = True

            if input_size is None:
                input_size = im_sz**2
                # self.f = NeuralNet(im_sz**2, hidden_units, extra_layers=args.nn_extra_layers, use_vbnorm=use_vbnorm, ref_x=ref_x, n_channels_in=args.n_ch)
            # else:
            self.f = NeuralNet(input_size, hidden_units, extra_layers=args.nn_extra_layers, use_vbnorm=use_vbnorm, ref_x=ref_x, n_channels_in=args.n_ch)
            self.f.last_dim = hidden_units
        else:
            self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate, input_channels=args.n_ch)

        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)
        print(self.f)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def penult(self, x):
        penult_z = self.f(x)
        return penult_z

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, im_sz=32,
                 use_nn=False, input_size=None, n_classes=10, ref_x=None, use_cnn=False):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate,
                                  n_classes=n_classes, im_sz=im_sz, input_size=input_size,
                                  use_nn=use_nn, ref_x=ref_x, use_cnn=use_cnn)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return t.gather(logits, 1, y[:, None])


def cond_entropy(logits):
    probs = t.softmax(logits, dim=1)
    # Use log softmax for stability.
    return - t.sum(probs * t.log_softmax(logits, dim=1)) / probs.shape[0]


def cycle(loader):
    while True:
        for data in loader:
            yield data


class FastLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.dataset_size = self.dataset[0].size(0)
        self.batch_size = batch_size

    def __next__(self):
        all_inds = np.array(list(range(self.dataset_size)))
        inds = np.random.choice(all_inds, self.batch_size, replace=False)
        inds = t.from_numpy(inds).long()
        batch = self.dataset[0][inds], self.dataset[1][inds] # data, labels
        return batch


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def init_random(args, bs):
    if (args.dataset == "moons" or args.dataset == "rings") or args.dataset in REG_DSETS:
        out = t.FloatTensor(bs, args.input_size).uniform_(-1,1) / args.temper_init
    elif args.dataset == "mnist":
        out = t.FloatTensor(bs, args.n_ch, args.im_sz, args.im_sz).uniform_(-3, 3) / args.temper_init
    else:
        out = t.FloatTensor(bs, args.n_ch, args.im_sz, args.im_sz).uniform_(-1, 1) / args.temper_init
    return out


def get_model_and_buffer(args, device, ref_x=None):
    model_cls = F if args.uncond else CCF
    args.input_size = None
    if args.dataset == "mnist":
        args.data_size = (1, 28, 28)
    elif args.dataset == "svhn" or args.dataset == "cifar10":
        args.data_dim = 32 * 32 * 3
        args.data_size = (3, 32, 32)

    if (args.dataset == "moons" or args.dataset == "rings"):
        args.input_size = 2
    if args.dataset in REG_DSETS:
        args.input_size = REG_DSETS[args.dataset]
        assert args.use_nn
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate,
                  n_classes=args.n_classes, im_sz=args.im_sz, input_size=args.input_size,
                  use_nn=args.use_nn, ref_x=ref_x, use_cnn=args.use_cnn)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
        f.load_state_dict(ckpt_dict["model_state_dict"])
        replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)

    if args.optim_sgld:
        replay_buffer = nn.Parameter(replay_buffer)

    return f, replay_buffer


def get_model_and_buffer_with_momentum(args, device, ref_x=None):
    f, replay_buffer = get_model_and_buffer(args, device, ref_x)
    momentum_buffer = t.zeros_like(replay_buffer)
    return f, replay_buffer, momentum_buffer


def logit_transform(x, clipping=0.05):
    precision = args.dequant_precision
    assert precision >= 2.0
    x = (x * (precision - 1) + t.rand_like(x)) / precision # noise for smoothness
    x = clipping + (1 - 2.0 * clipping) * x # clipping to avoid explosion at ends
    x = t.log(x) - t.log(1.0 - x)
    return x


def get_data(args):
    if args.dataset == "svhn":
        if args.svhn_logit_transform:
            transform_train = tr.Compose(
                [tr.Pad(4),
                 tr.RandomCrop(args.im_sz),
                 tr.ToTensor(),
                 logit_transform,
                 lambda x: x + args.sigma * t.randn_like(x)
                 ]
            )
        else:
            transform_train = tr.Compose(
                [tr.Pad(4, padding_mode="reflect"),
                 tr.RandomCrop(args.im_sz),
                 tr.ToTensor(),
                 tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                 lambda x: x + args.sigma * t.randn_like(x)]
            )
    elif args.dataset == "mnist":
        if args.mnist_no_logit_transform:
            transform_train = tr.Compose(
                [tr.ToTensor()]
            )
        elif args.mnist_no_crop:
            transform_train = tr.Compose(
                [
                 tr.ToTensor(),
                 logit_transform,
                 lambda x: x + args.mnist_sigma * t.randn_like(x)
                 ]
            )
        else:
            transform_train = tr.Compose(
                [tr.Pad(4),
                 tr.RandomCrop(args.im_sz),
                 tr.ToTensor(),
                 logit_transform,
                 lambda x: x + args.mnist_sigma * t.randn_like(x)
                 ]
            )
    elif (args.dataset == "moons" or args.dataset == "rings"):
        transform_train = None
    elif args.dataset in REG_DSETS:
        transform_train = None
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(args.im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    if args.dataset == "mnist":
        if args.mnist_no_logit_transform:
            transform_test = tr.Compose(
                [tr.ToTensor()]
            )
        else:
            transform_test = tr.Compose(
                [tr.ToTensor(),
                 logit_transform]
            )
    elif (args.dataset == "moons" or args.dataset == "rings"):
        transform_test = None
    elif args.dataset in REG_DSETS:
        transform_test = None
    else:
        transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "mnist":
            return tv.datasets.MNIST(root=args.data_root, transform=transform, download=True, train=train)
        elif (args.dataset == "moons" or args.dataset == "rings"):
            if args.dataset == "moons":
                data, labels = datasets.make_moons(n_samples=args.n_moons_data, noise=args.moons_noise, random_state=np.random.RandomState(args.data_seed))
            elif args.dataset == "rings":
                data, labels = datasets.make_circles(n_samples=args.n_rings_data, noise=args.rings_noise, random_state=np.random.RandomState(args.data_seed))

            data = t.Tensor(data)

            labels = t.Tensor(labels)
            labels = labels.long()
            return t.utils.data.TensorDataset(data, labels)
        elif args.dataset in REG_DSETS:
            tr, te, ddim = regression_datasets.get_data(args.dataset, seed=args.data_seed)
            if train:
                return tr
            else:
                return te

        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")


    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(args.dataset_seed)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []

    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            #
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)

    if args.dataset in TOY_DSETS or args.dataset in REG_DSETS:
        dset_train_labeled = dataset_fn(True, transform_train)[train_labeled_inds]
        dload_train_labeled = FastLoader(dset_train_labeled,
                                         batch_size=min(args.batch_size, len(
                                             dset_train_labeled[0])))
    else:
        dset_train_labeled = DataSubset(
            dataset_fn(True, transform_train),
            inds=train_labeled_inds)
        dload_train_labeled = DataLoader(dset_train_labeled, batch_size=min(args.batch_size, len(dset_train_labeled)), shuffle=True, num_workers=4, drop_last=True)
        dload_train_labeled = cycle(dload_train_labeled)

    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.ul_batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_vbnorm = DataLoader(dset_train, batch_size=args.vbnorm_batch_size, shuffle=False, num_workers=4, drop_last=True)


    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    dload_train_labeled_static = DataLoader(dset_train_labeled, batch_size=min(args.batch_size, len(dset_train_labeled)), shuffle=False, num_workers=4, drop_last=True)
    dload_train_labeled_static = cycle(dload_train_labeled_static)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid,dload_test, dset_train, dset_train_labeled, dload_train_labeled_static, dload_train_vbnorm


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None, momentum_buffer=None, data=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        if (args.dataset == "moons" or args.dataset == "rings") or args.dataset in REG_DSETS:
            choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None]
        else:
            choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        if args.buffer_reinit_from_data:
            assert data is not None
            samples = choose_random * data.cpu() + (1 - choose_random) * buffer_samples
        else:
            samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        if momentum_buffer is not None:
            momentum_buffer[inds] *= (1-choose_random) # Reset momentum to 0 when resetting data, keep as before if choosing buffer sample
        return samples.to(device), inds

    def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps, seed_batch=None,
                 optim_sgld=None, momentum_buffer=None, data=None):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.ul_batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        if seed_batch is not None:
            init_sample, buffer_inds = seed_batch, []
        else:
            init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y,
                                                  momentum_buffer=momentum_buffer, data=data)
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        if momentum_buffer is not None:
            momentum = momentum_buffer[buffer_inds].to(device)
        fp_sum2 = 0
        fp_sum = 0
        fp_count = 0
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]

            fp_sum2 += (f_prime**2).sum().item()
            fp_sum += f_prime.sum().item()
            fp_count += np.prod(f_prime.shape)

            if momentum_buffer is not None:
                # Modification to usual momentum to "conserve energy" which should help for sampling
                momentum = (args.sgld_momentum * momentum + (1-args.sgld_momentum) * f_prime)
                x_k.data += args.sgld_lr * momentum
            else:
                # import pdb; pdb.set_trace()
                if args.sgld_rmsp > 0:
                    fp_std = t.max(t.std(f_prime.view(f_prime.shape[0], -1), 1),
                                   t.tensor(args.sgld_rmsp, device=f_prime.device)) / args.sgld_rmsp
                    fp_std = fp_std.view([-1] + [1] * (f_prime.dim()-1))
                    x_k.data += args.sgld_lr * f_prime / fp_std
                else:
                    x_k.data += args.sgld_lr * f_prime
                x_k.data += args.sgld_std * t.randn_like(x_k)

                if args.sgld_clamp:
                    x_k.data = t.clamp(x_k.data, -args.sgld_clamp, args.sgld_clamp)
        f_prime_std = np.sqrt(fp_sum2 / fp_count - (fp_sum / fp_count) ** 2)
        print(f'f_prime_std: {f_prime_std}')

        f.train()
        if args.optim_sgld:
            final_samples = replay_buffer[buffer_inds].to(device).detach()
        else:
            final_samples = x_k.detach()
        if momentum_buffer is not None:
            momentum_buffer[buffer_inds] = momentum.cpu()

        # update replay buffer
        if seed_batch is None:
            # Only update replay buffer in PCD (CD = use seed batch at data)
            # Just detaching functionality for now
            if len(replay_buffer) > 0:
                if args.optim_sgld:
                    replay_buffer[buffer_inds].data = final_samples.cpu()
                else:
                    replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples

    return sample_q


def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)

def plot_jacobian_spectrum(x_samples, f, epoch, use_penult=False):
    for c in range(args.n_classes):
        x_example = x_samples[c]
        x_example.requires_grad = True
        j_list = []
        f.eval()
        # Is the below Jacobian calculation vectorizable?
        dim = args.n_classes
        if use_penult:
            dim = f.penult(x_example).squeeze().shape[0]
            penult_plot_num = 20
        for i in range(dim):
            if use_penult:
                grad = t.autograd.grad(f.penult(x_example).squeeze()[i],
                                       x_example)[0]
            else:
                grad = t.autograd.grad(f.classify(x_example)[i],
                                       x_example)[0]
            grad = grad.reshape(-1)
            j_list.append(grad)
        f.train()
        jacobian = t.stack(j_list)
        u, s, v = t.svd(jacobian)
        spectrum = s.detach().cpu().numpy()
        if use_penult:
            plt.scatter(np.arange(0, penult_plot_num), spectrum[0:penult_plot_num])
            fig_name = "spectrum_digit{}_epoch{}_penult".format(c, epoch)
        else:
            plt.scatter(np.arange(0, args.n_classes), spectrum)
            fig_name = "spectrum_digit{}_epoch{}".format(c, epoch)
        plt.savefig(fig_name)
        plt.close()

def decision_boundary(net, X, device, plt_bndry=0.5):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - plt_bndry, X[:, 0].max() + plt_bndry
    y_min, y_max = X[:, 1].min() - plt_bndry, X[:, 1].max() + plt_bndry
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    xxt = t.from_numpy(xx.ravel()).float()
    yyt = t.from_numpy(yy.ravel()).float()
    xxyy = t.cat([xxt[:, None], yyt[:, None]], dim=1)
    logits = net.classify(xxyy.to(device))
    Z = logits.argmax(1)
    plt.pcolormesh(xx, yy, Z.cpu().numpy().reshape(xx.shape), cmap=ListedColormap(['r', 'b']), alpha=.1)

def visualize_decision_boundary(net, Xf, Xfl, device, fname="data.png"):
    plt.clf()
    plt.axis('off')
    decision_boundary(net, Xf, device)
    plt.scatter(Xf[:, 0], Xf[:, 1], c='grey', alpha=0.3)
    plt.scatter(Xfl[:args.labels_per_class, 0], Xfl[:args.labels_per_class, 1], c='r')
    plt.scatter(Xfl[args.labels_per_class:, 0], Xfl[args.labels_per_class:, 1], c='b')
    plt.savefig("{}/{}".format(args.save_dir, fname))


def main(args):
#     import pydevd_pycharm
#     pydevd_pycharm.settrace('cymes.stud.ii', port=7321, stdoutToServer=True, stderrToServer=True)

    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(args.t_seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.t_seed)

    if args.dataset == "mnist":
        args.n_ch = 1
        args.im_sz = 28
    elif (args.dataset == "moons" or args.dataset == "rings"):
        args.n_ch = 1
        args.im_sz = None
    elif args.dataset in REG_DSETS:
        args.n_ch = 1
        args.im_sz = None
    else:
        args.n_ch = 3
        args.im_sz = 32

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test, dset_train, \
    dset_train_labeled, dload_train_labeled_static, dload_train_vbnorm = get_data(args)
    print('Train labeled indexes', dset_train_labeled.inds[:10])

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    ref_x = None
    if args.vbnorm:
        ref_x = next(iter(dload_train_vbnorm))[0].to(device)

    sample_q = get_sample_q(args, device)

    momentum_buffer = None
    if args.use_sgld_momentum:
        f, replay_buffer, momentum_buffer = get_model_and_buffer_with_momentum(args, device, ref_x)
    else:
        f, replay_buffer = get_model_and_buffer(args, device, ref_x)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    optim_sgld = None
    if args.optim_sgld:
        # This SGD optimizer is basically SGLD with 0 noise
        optim_sgld = t.optim.SGD([replay_buffer], lr=args.sgld_lr, momentum=args.optim_sgld_momentum)


    best_valid_acc = 0.0
    cur_iter = 0

    if args.svd_jacobian:
        # Collect static samples for consistent evaluation of SVD of Jacobian
        # 1 static sample per class for now
        # zero init at first
        static_samples = init_random(args, args.n_classes).to(device) * 0
        count = 0
        # Assumes we have an instance of every class (might have infinite loop if we don't)
        for i, (x_lab, y_lab) in enumerate(dload_train_labeled_static):
            for j in range(len(y_lab)):
                if static_samples[y_lab[j]].sum() == 0:
                    static_samples[y_lab[j]] = x_lab[j]
                    count += 1
            # Stop when we have all classes
            if count == args.n_classes:
                break


    if args.eval_mode_except_clf:
        f.eval()


    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))


        for i, (x_p_d, _) in tqdm(enumerate(dload_train)):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            seed_batch = None
            if args.use_cd:
                seed_batch = x_p_d.clone() # breaks reference, important

            L = 0.

            if args.vat:

                if args.eval_mode_except_clf:
                    f.train()

                optim.zero_grad()
                vat_loss = VATLoss(xi=10.0, eps=args.vat_eps, ip=1)
                lds = vat_loss(f, x_p_d)

                logits = f.classify(x_lab)

                loss = args.p_y_given_x_weight * nn.CrossEntropyLoss()(logits, y_lab) + args.vat_weight * lds

                if args.ent_min:
                    # loss += cond_entropy(logits) * args.ent_min_weight
                    logits_unlab = f.classify(x_p_d)
                    loss += cond_entropy(logits_unlab) * args.ent_min_weight


                loss.backward()
                optim.step()

                if args.eval_mode_except_clf:
                    f.eval()

                cur_iter += 1

                if cur_iter % args.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print(
                        'P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(
                            epoch,
                            cur_iter,
                            loss.item(),
                            acc.item()))

                if args.svd_jacobian and cur_iter % args.svd_every == 0:
                    plot_jacobian_spectrum(static_samples, f, epoch)
                    plot_jacobian_spectrum(static_samples, f, epoch,
                                           use_penult=True)

            else:
                l_p_x = t.tensor(0.0)
                if args.p_x_weight > 0:  # maximize log p(x)
                    if args.score_match:
                        sm_loss = sliced_score_matching(f, x_p_d, args.n_sm_vectors)
                        L += args.p_x_weight * sm_loss
                        if cur_iter % args.print_every == 0:
                            print('sm_loss {}:{:>d} = {:>14.9f}'.format(
                                    epoch, i, sm_loss))
                    elif args.denoising_score_match:
                        # Multiply by args.denoising_sm_sigma**2 to keep scale of loss
                        # constant across sigma changes
                        # See 4.2 in Generative Modeling by Estimating Gradients of the
                        # Data Distribution (Yang, Ermon 2019)
                        sm_loss = args.denoising_sm_sigma**2 * denoising_score_matching(f, x_p_d,
                                                        args.denoising_sm_sigma)
                        L += args.p_x_weight * sm_loss
                        if cur_iter % args.print_every == 0:
                            print('sm_loss {}:{:>d} = {:>14.9f}'.format(
                                epoch, i, sm_loss))

                    else:
                        # else:
                        if args.class_cond_p_x_sample:
                            assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                            y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                            x_q = sample_q(f, replay_buffer, y=y_q, optim_sgld=optim_sgld,
                                           seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)

                        else:
                            x_q = sample_q(f, replay_buffer, optim_sgld=optim_sgld,
                                           seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)  # sample from log-sumexp

                        fp_all = f(x_p_d)
                        fq_all = f(x_q)
                        fp = fp_all.mean()
                        fq = fq_all.mean()

                        l_p_x = -(fp - fq)
                        if cur_iter % args.print_every == 0:
                            print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                           fp - fq))
                        L += args.p_x_weight * l_p_x

                        if args.l2_energy_reg > 0:
                            L += args.l2_energy_reg * (fp_all ** 2).sum()
                            # If we want to regularize the negative samples as https://arxiv.org/pdf/1903.08689.pdf does too
                            if args.l2_energy_reg_neg:
                                L += args.l2_energy_reg * (fq_all ** 2).sum()


                if args.p_y_given_x_weight > 0:  # maximize log p(y | x)
                    if args.eval_mode_except_clf:
                        f.train()

                    logits = f.classify(x_lab)
                    l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)

                    if cur_iter % args.print_every == 0:
                        acc = (logits.max(1)[1] == y_lab).float().mean()
                        print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                                     cur_iter,
                                                                                     l_p_y_given_x.item(),
                                                                                     acc.item()))

                    if args.svd_jacobian and cur_iter % args.svd_every == 0:
                        plot_jacobian_spectrum(static_samples, f, epoch)
                        plot_jacobian_spectrum(static_samples, f, epoch,
                                               use_penult=True)

                    L += args.p_y_given_x_weight * l_p_y_given_x

                    if args.ent_min:
                        # L += cond_entropy(logits) * args.ent_min_weight
                        logits_unlab = f.classify(x_p_d)
                        # Just unlabeled now
                        L += cond_entropy(logits_unlab) * args.ent_min_weight

                if args.p_x_y_weight > 0:  # maximize log p(x, y)
                    assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                    x_q_lab = sample_q(f, replay_buffer, y=y_lab, optim_sgld=optim_sgld,
                                       seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)
                    fp, fq = f(x_lab, y_lab).mean(), f(x_q_lab, y_lab).mean()
                    l_p_x_y = -(fp - fq)
                    if cur_iter % args.print_every == 0:
                        print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                          fp - fq))

                    L += args.p_x_y_weight * l_p_x_y

                if args.vat_also:
                    vat_loss = VATLoss(xi=10.0, eps=args.vat_eps, ip=1)
                    lds = vat_loss(f, x_p_d)
                    L += args.vat_also_weight * lds

                if args.class_cond_label_prop and cur_iter > args.warmup_iters:

                    lds_loss = LDSLoss(n_steps=args.label_prop_n_steps)
                    lds = lds_loss(f, x_p_d, sample_q, seed_batch=x_p_d)

                    L += args.label_prop_weight * lds

                # break if the loss diverged...easier for poppa to run experiments this way
                if L.abs().item() > 1e8:
                    print("BAD BOIIIIIIIIII")
                    1/0

                optim.zero_grad()
                L.backward()
                
                assert len(optim.param_groups) == 1
                params = optim.param_groups[0]['params']
                gn = nn.utils.clip_grad_norm_(params, args.grad_clip)
                # import pdb; pdb.set_trace()
                if cur_iter % min(10, args.print_every) == 0:
                    print (f'L: {L.item()}, gn: {gn}, l_p_x: {l_p_x.item()}')

                optim.step()    
                
                if args.eval_mode_except_clf:
                    f.eval()

                cur_iter += 1

                if cur_iter % args.viz_every == 0:
                    if args.plot_uncond:
                        if args.class_cond_p_x_sample:
                            assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                            y_q = t.randint(0, args.n_classes, (args.ul_batch_size,)).to(device)
                            x_q = sample_q(f, replay_buffer, y=y_q, optim_sgld=optim_sgld,
                                           seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)
                        else:
                            x_q = sample_q(f, replay_buffer, optim_sgld=optim_sgld,
                                           seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)
                        if args.dataset == "moons" or args.dataset == "rings":
                            d = x_q.detach().cpu().numpy()
                            plt.clf()
                            plt.scatter(d[:, 0], d[:, 1], c="orange")
                            plt.savefig('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i))
                        else:
                            plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                    if args.plot_cond:  # generate class-conditional samples
                        y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                        x_q_y = sample_q(f, replay_buffer, y=y, optim_sgld=optim_sgld,
                                         seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)
                        plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)

        if epoch % args.ckpt_every == 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0):
            f.eval()
            with t.no_grad():
                # validation set
                best_valid_found = False
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct >= best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    best_valid_found = True
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)


                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            f.train()

            if (args.dataset == "moons" or args.dataset == "rings") and best_valid_found:
                def vis(savefile, random_state=None):
                    plt.clf()
                    if args.dataset == "moons":
                        data, labels = datasets.make_moons(args.n_moons_data, noise=args.moons_noise, random_state=random_state)
                    elif args.dataset == "rings":
                        data, labels = datasets.make_circles(
                            n_samples=args.n_rings_data, noise=args.rings_noise,
                            random_state=np.random.RandomState(
                                args.data_seed))
                    data = t.Tensor(data)
                    # data = data.to(device)

                    labeled_pts = dset_train_labeled[:][0]
                    print("Saving figure")

                    visualize_decision_boundary(f, data, labeled_pts, device, savefile)

                if args.dataset == "moons":
                    vis(savefile="{}/moonvis_train.png".format(args.save_dir), random_state=np.random.RandomState(args.data_seed))
                    vis(savefile="{}/moonvis_test.png".format(args.save_dir))
                elif args.dataset == "rings":
                    vis(savefile="{}/ringsvis_train.png".format(args.save_dir),
                        random_state=np.random.RandomState(
                            args.data_seed))
                    vis(savefile="{}/ringsvis_test.png".format(args.save_dir))

        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="moons", choices=["cifar10", "svhn", "mnist",
                                                                         "cifar100", "moons", "rings",
                                                                         "concrete", "protein", "navy",
                                                                         "power_plant", "year"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="Note: use p_x_weight 0.0 instead of this for baseline classifier. If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=10,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--ul_batch_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--label_prop_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--mnist_sigma", type=float, default=0.0,
                        help="stddev of gaussian noise to add to input for mnist, after logit transform")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)
    # parser.add_argument("--semi-supervised", type=bool, default=False)
    parser.add_argument("--vat", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--vat_weight", type=float, default=1.0)
    parser.add_argument("--n_moons_data", type=int, default=1000, help="how many data points in moon dataset")
    parser.add_argument("--moons_noise", type=float, default=0.1, help="how much noise to add to moons dataset")
    parser.add_argument("--data_seed", type=int, default=1234, help="for training dataset")
    parser.add_argument("--n_rings_data", type=int, default=1000,
                        help="how many data points in rings dataset")
    parser.add_argument("--rings_noise", type=float, default=0.03,
                        help="how much noise to add to rings dataset")
    parser.add_argument("--class_cond_label_prop", action="store_true", help="Enforce consistency/LDS between data and samples too")
    parser.add_argument("--label_prop_n_steps", type=int, default=1,
                        help="number of steps of SGLD sampler for label prop idea")
    parser.add_argument("--svd_jacobian", action="store_true", help="Do SVD on Jacobian matrix at data points to help understand model behaviour")
    parser.add_argument("--svd_every", type=int, default=300, help="Iterations between svd")
    parser.add_argument("--vat_eps", type=float, default=3.0)
    parser.add_argument("--vat_also_weight", type=float, default=1.0)
    parser.add_argument("--vat_also", action="store_true", help="Run VAT together with JEM")
    parser.add_argument("--ent_min", action="store_true", help="Run With Entropy Minimization")
    parser.add_argument("--ent_min_weight", type=float, default=0.1)
    parser.add_argument("--vbnorm", action="store_true", help="Run with Virtual Batch Norm")
    parser.add_argument("--vbnorm_batch_size", type=int, default=1000)
    parser.add_argument("--batch_norm", action="store_true", help="Run with Batch Norm (on NN; CNN has by default)")
    parser.add_argument("--dropout", action="store_true", help="Run with Dropout (on NN; CNN has by default)")
    parser.add_argument("--mnist_no_logit_transform", action="store_true", help="Run MNIST without logit transform")
    parser.add_argument("--mnist_no_crop", action="store_true", help="Run MNIST without crop")
    parser.add_argument("--score_match", action="store_true", help="Note: so far implemented only for p(x). Use score matching instead of SGLD in training JEM")
    parser.add_argument("--swish", action="store_true", help="Use swish activation on NN instead of ReLU")
    parser.add_argument("--softplus", action="store_true", help="Use softplus activation on NN instead of ReLU")
    parser.add_argument("--sine", action="store_true", help="Use sine activation on NN instead of ReLU")
    parser.add_argument("--n_sm_vectors", type=int, default=1, help="Number of vectors for projection with score matching")
    parser.add_argument("--no_param_bn", action="store_true", help="No affine transform/learnable BN params")
    parser.add_argument("--first_layer_bn_only", action="store_true")
    parser.add_argument("--dequant_precision", type=float, default=256.0, help="For dequantization/logit transform")
    parser.add_argument("--denoising_score_match", action="store_true", help="Use denoising score matching to train")
    parser.add_argument("--denoising_sm_sigma", type=float, default=0.1, help="Noise to add in denoising score matching")
    parser.add_argument("--leaky_relu", action="store_true", help="Use Leaky ReLU activation on NN instead of ReLU. Note CNN has leaky ReLU by default")
    parser.add_argument("--eval_mode_except_clf", action="store_true", help="Pytorch eval mode on everything except classifier training")
    parser.add_argument("--use_cnn", action="store_true", help="Use CNN")
    parser.add_argument("--cnn_no_bn", action="store_true", help="No BN on CNN architecture")
    parser.add_argument("--cnn_no_dropout", action="store_true", help="No Dropout on CNN architecture")
    parser.add_argument("--optim_sgld", action="store_true", help="Use SGLD Optimizer")
    parser.add_argument("--optim_sgld_momentum", type=float, default=0.0)
    parser.add_argument("--use_cd", action="store_true", help="Use contrastive divergence instead of persistent contrastive divergence (initialize from data instead of saved replay buffer/previous samples")
    parser.add_argument("--svhn_logit_transform", action="store_true", help="Run SVHN with logit transform")
    parser.add_argument("--use_sgld_momentum", action="store_true")
    parser.add_argument("--sgld_momentum", type=float, default=0.9)
    parser.add_argument("--cnn_avg_pool_kernel", type=int, default=6)
    parser.add_argument("--use_nn", action="store_true", help="Use NN (4 layer MLP)")
    parser.add_argument("--nn_hidden_size", type=int, default=500)
    parser.add_argument("--nn_extra_layers", type=int, default=2, help="2 for a 4-layer MLP (this is only middle layers, excludes input and output layers)")
    parser.add_argument("--buffer_reinit_from_data", action="store_true", help="For PCD replay buffer, reinitialize from data points rather than random points")
    parser.add_argument("--h_dim", type=int, default=100)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--p_control", type=float, default=0.0)
    parser.add_argument("--log_sigma_low", type=float, default=.001)
    parser.add_argument("--log_sigma_high", type=float, default=0.2)
    parser.add_argument("--viz_every", type=int, default=100, help="Iterations between visualization")
    parser.add_argument("--resnet", action="store_true", help="Resnet for PGAN")
    parser.add_argument("--l2_energy_reg", type=float, default=0., help="Regularize energy outputs")
    parser.add_argument("--l2_energy_reg_neg", action="store_true", help="Regularize energy outputs on negative samples (x_q) as well")
    parser.add_argument("--dataset_seed", type=int, default=1234, help="for selecting data")
    parser.add_argument("--t_seed", type=int, default=1, help="for Torch")
    parser.add_argument("--temper_init", type=float, default=1.0, help="Reduces the range of the initial uniform dist, may allow for more stable sampling")
    parser.add_argument("--sm_dim", type=int, default=128, help="Dimension of the SoftMax layer")
    parser.add_argument("--grad_clip", type=float, default=1e10, help="gradient clipping")
    parser.add_argument("--sgld_clamp", type=float, default=0.0, help="clamp SGLD samples")
    parser.add_argument("--sgld_rmsp",  type=float, default=0.0, help="normalize SGLD gradients")
    parser.add_argument("--spectral_norm", action="store_true", default=False, help="apply spectral-norm to model.")


    args = parser.parse_args()
    if args.dataset == "cifar100":
        args.n_classes = 100
    elif (args.dataset == "moons" or args.dataset == "rings"):
        args.n_classes = 2
    elif args.dataset in REG_DSETS or args.dataset == "mnist" or args.dataset == "svhn" or args.dataset == "cifar10":
        args.n_classes = 10
    if args.vat:
        print("Running VAT")


    main(args)
