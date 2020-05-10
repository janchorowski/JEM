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
seed = 1
# im_sz = 32
# n_ch = 3
from sklearn import datasets
import matplotlib.pyplot as plt
from vbnorm import VirtualBatchNormNN
# from batch_renormalization import BatchRenormalizationNN
from batchrenorm import BatchRenorm1d
from losses import VATLoss, LDSLoss, sliced_score_matching_vr, \
    sliced_score_matching, denoising_score_matching


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


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, extra_layers=2, use_vbnorm=False, ref_x=None, n_channels_in=1):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        self.use_vbnorm = use_vbnorm

        self.n_channels_in = n_channels_in

        affine = True
        if args.no_param_bn:
            affine = False

        layer_in = nn.Linear(input_size * n_channels_in, hidden_size)
        self.layers.append(layer_in)
        self.ref_x = ref_x

        if use_vbnorm:
            assert ref_x is not None
            self.layers.append(VirtualBatchNormNN(hidden_size))
            # self.layers.append(BatchRenormalizationNN(hidden_size))
            # self.layers.append(BatchRenorm1d(hidden_size))
        elif args.batch_norm:
            self.layers.append(nn.BatchNorm1d(num_features=hidden_size, affine=affine))
        if args.swish:
            self.layers.append(Swish(hidden_size))
        elif args.leaky_relu:
            self.layers.append(nn.LeakyReLU())
        else:
            self.layers.append(nn.ReLU())


        for i in range(extra_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            if not args.first_layer_bn_only:
                if use_vbnorm:
                    self.layers.append(VirtualBatchNormNN(hidden_size))
                    # self.layers.append(VirtualBatchNormNN(hidden_size))
                    # self.layers.append(BatchRenorm1d(hidden_size))
                elif args.batch_norm:
                    self.layers.append(nn.BatchNorm1d(num_features=hidden_size, affine=affine))
            if args.swish:
                self.layers.append(Swish(hidden_size))
            elif args.leaky_relu:
                self.layers.append(nn.LeakyReLU())
            else:
                self.layers.append(nn.ReLU())

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
            elif isinstance(layer, BatchRenorm1d) or isinstance(layer, nn.BatchNorm1d):
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


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, im_sz=32, use_nn=False, input_size=None, n_classes=10, ref_x=None, use_cnn=False):
        if input_size is not None:
            assert use_nn == True #input size is for non-images, ie non-conv.
        super(F, self).__init__()

        if use_cnn:
            print("Using ConvLarge")
            self.f = ConvLarge(avg_pool_kernel=args.cnn_avg_pool_kernel)
            self.f.last_dim = 128
        elif use_nn:
            hidden_units = 500

            use_vbnorm = False
            if args.vbnorm:
                use_vbnorm = True

            if input_size is None:
                self.f = NeuralNet(im_sz**2, hidden_units, extra_layers=2, use_vbnorm=use_vbnorm, ref_x=ref_x, n_channels_in=args.n_ch)
            else:
                self.f = NeuralNet(input_size, hidden_units, extra_layers=2, use_vbnorm=use_vbnorm, ref_x=ref_x, n_channels_in=args.n_ch)
            self.f.last_dim = hidden_units
        else:
            self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate, input_channels=args.n_ch)

        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

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
    if args.dataset == "moons":
        out = t.FloatTensor(bs, args.input_size).uniform_(-1,1)
    else:
        out = t.FloatTensor(bs, args.n_ch, args.im_sz, args.im_sz).uniform_(-1, 1)
    return out


def get_model_and_buffer(args, device, sample_q, ref_x=None):
    model_cls = F if args.uncond else CCF
    args.input_size = None
    use_nn = args.use_nn
    # if args.dataset == "mnist" or args.dataset == "moons":
        # use_nn=False # testing only
    if args.dataset == "mnist":
        args.data_size = (1, 28, 28)

    if args.dataset == "moons":
        args.input_size = 2
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate,
                  n_classes=args.n_classes, im_sz=args.im_sz, input_size=args.input_size,
                  use_nn=use_nn, ref_x=ref_x, use_cnn=args.use_cnn)
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


def get_model_and_buffer_with_momentum(args, device, sample_q, ref_x=None):
    f, replay_buffer = get_model_and_buffer(args, device, sample_q, ref_x)
    momentum_buffer = t.zeros_like(replay_buffer)
    return f, replay_buffer, momentum_buffer




def logit_transform(x, lamb = 0.05):
    # Adapted from https://github.com/yookoon/VLAE
    # x = (x * 255.0 + t.rand_like(x)) / 256.0 # noise
    precision = args.dequant_precision
    assert precision >= 2.0
    x = (x * (precision - 1) + t.rand_like(x)) / precision # noise for smoothness
    x = lamb + (1 - 2.0 * lamb) * x # clipping to avoid explosion at ends
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
        if args.pgan:
            transform_train = tr.Compose([tr.ToTensor(),
                                            lambda x: (((255. * x) + t.rand_like(
                                                x)) / 256.)])
        elif args.mnist_no_logit_transform:
            transform_train = tr.Compose(
                [
                 # tr.Pad(4),
                 # tr.RandomCrop(args.im_sz),
                 tr.ToTensor(),
                 # lambda x: x + args.mnist_sigma * t.randn_like(x)
                 ]
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
    elif args.dataset == "moons":
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
        if args.pgan:
            transform_test = tr.Compose([tr.ToTensor(),])
        elif args.mnist_no_logit_transform:
            transform_test = tr.Compose(
                [tr.ToTensor(),
                 # tr.Normalize((.5,), (.5,)),
                 # lambda x: x + args.sigma * t.randn_like(x)
                 # lambda x: x + args.mnist_sigma * t.randn_like(x)
                ]
            )
        elif args.mnist_no_crop:
            transform_test = tr.Compose(
                [tr.ToTensor(),
                 logit_transform,
                 ]
            )
        else:
            transform_test = tr.Compose(
                [tr.ToTensor(),
                 # tr.Normalize((.5,), (.5,)),
                 # lambda x: x + args.sigma * t.randn_like(x)
                 logit_transform,
                 # lambda x: x + args.mnist_sigma * t.randn_like(x)
                 ]
            )
    elif args.dataset == "moons":
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
        elif args.dataset == "moons":
            data,labels = datasets.make_moons(n_samples=args.n_moons_data, noise=.1)

            # plt.scatter(data[:,0],data[:,1])
            # plt.show()
            data = t.Tensor(data)

            labels = t.Tensor(labels)
            labels = labels.long()
            return t.utils.data.TensorDataset(data, labels)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")


    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
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
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_vbnorm = DataLoader(dset_train, batch_size=args.vbnorm_batch_size, shuffle=False, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dload_train_labeled_static = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
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
        if args.dataset == "moons":
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
        bs = args.batch_size if y is None else y.size(0)
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
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            # print(t.sum(t.abs(f_prime)))
            # Note f_prime is log sum exp whereas our energy function is neg log sum exp
            # So the reason our steps were positive before was it was minus a negative
            neg_f_prime = -f_prime
            # This negative f prime is the gradient of the energy function, which we are taking steps
            # in descent with respect to.
            # x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
            if momentum_buffer is not None:
                # Modification to usual momentum to "conserve energy" which should help for sampling
                momentum = (args.sgld_momentum * momentum + (1-args.sgld_momentum) * f_prime)
                x_k.data += args.sgld_lr * momentum
                # No noise with momentum right now but can do so if we want by
                # unindenting the line 3 lines below
            else:
                # old = x_k.data + 0.0 # +0.0 breaks a reference, so it's now a copy instead of a reference
                # new = x_k.data + args.sgld_lr * f_prime
                x_k.data += args.sgld_lr * f_prime
                # print("---")
                # print(args.sgld_lr * f_prime)
                # print(x_k.data)
                # new2 = x_k.data
                # print(t.sum(t.abs(args.sgld_lr * f_prime)))
                # print(t.sum(t.abs(old)-t.abs(new)))
                # print(t.sum(t.abs(old)-t.abs(new2)))
                x_k.data += args.sgld_std * t.randn_like(x_k)

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
    # Let's just do 1 example for now
    # input_ex_ind = 0
    # x_example = x_lab[input_ex_ind]
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
        # print(s)
        spectrum = s.detach().cpu().numpy()
        if use_penult:
            plt.scatter(np.arange(0, penult_plot_num), spectrum[0:penult_plot_num])
            fig_name = "spectrum_digit{}_epoch{}_penult".format(c, epoch)
        else:
            plt.scatter(np.arange(0, args.n_classes), spectrum)
            fig_name = "spectrum_digit{}_epoch{}".format(c, epoch)
        plt.savefig(fig_name)
        # plt.show()
        plt.close()


def main(args):

    if args.pgan_test:

        import torch
        import torch.nn as nn
        import torch.utils
        import numpy as np
        import torch.distributions as distributions
        from torch.utils.data import DataLoader, Dataset, TensorDataset
        from torchvision import datasets, transforms
        import torchvision
        import sklearn.datasets as skdatasets
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import utils
        import toy_data
        TOY_DSETS = ("moons", "circles", "8gaussians", "pinwheel", "2spirals",
                     "checkerboard", "rings", "swissroll")

        if args.dataset in TOY_DSETS:
            args.data_dim = 2
        elif args.dataset == "mnist":
            args.data_dim = 784
            args.data_size = (1, 28, 28)

        def logit(x, alpha=1e-6):
            x = x * (1 - 2 * alpha) + alpha
            return torch.log(x) - torch.log(1 - x)

        def _helper(netG, x_tilde, eps, sigma):
            eps = eps.clone().detach().requires_grad_(True)
            with torch.no_grad():
                G_eps = netG(eps)
            bsz = eps.size(0)
            log_prob_eps = (eps ** 2).view(bsz, -1).sum(1).view(-1, 1)
            log_prob_x = (x_tilde - G_eps) ** 2 / sigma ** 2
            log_prob_x = log_prob_x.view(bsz, -1)
            log_prob_x = torch.sum(log_prob_x, dim=1).view(-1, 1)
            logjoint_vect = -0.5 * (log_prob_eps + log_prob_x)
            logjoint_vect = logjoint_vect.squeeze()
            logjoint = torch.sum(logjoint_vect)
            logjoint.backward()
            grad_logjoint = eps.grad
            return logjoint_vect, logjoint, grad_logjoint

        def get_samples(netG, x_tilde, eps_init, sigma, burn_in,
                        num_samples_posterior,
                        leapfrog_steps, stepsize, flag_adapt, hmc_learning_rate,
                        hmc_opt_accept):
            device = eps_init.device
            bsz, eps_dim = eps_init.size(0), eps_init.size(1)
            n_steps = burn_in + num_samples_posterior
            acceptHist = torch.zeros(bsz, n_steps).to(device)
            logjointHist = torch.zeros(bsz, n_steps).to(device)
            samples = torch.zeros(bsz * num_samples_posterior, eps_dim).to(
                device)
            current_eps = eps_init
            cnt = 0
            for i in range(n_steps):
                eps = current_eps
                p = torch.randn_like(current_eps)
                current_p = p
                logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde,
                                                                 current_eps,
                                                                 sigma)
                current_U = -logjoint_vect.view(-1, 1)
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U / 2.0
                for j in range(leapfrog_steps):
                    eps = eps + stepsize * p
                    if j < leapfrog_steps - 1:
                        logjoint_vect, logjoint, grad_logjoint = _helper(netG,
                                                                         x_tilde,
                                                                         eps,
                                                                         sigma)
                        proposed_U = -logjoint_vect
                        grad_U = -grad_logjoint
                        p = p - stepsize * grad_U
                logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde,
                                                                 eps, sigma)
                proposed_U = -logjoint_vect.view(-1, 1)
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U / 2.0
                p = -p
                current_K = 0.5 * (current_p ** 2).sum(dim=1)
                current_K = current_K.view(-1, 1)  ## should be size of B x 1
                proposed_K = 0.5 * (p ** 2).sum(dim=1)
                proposed_K = proposed_K.view(-1, 1)  ## should be size of B x 1
                unif = torch.rand(bsz).view(-1, 1).to(device)
                accept = unif.lt(
                    torch.exp(current_U - proposed_U + current_K - proposed_K))
                accept = accept.float().squeeze()  ## should be B x 1
                acceptHist[:, i] = accept
                ind = accept.nonzero().squeeze()
                try:
                    len(ind) > 0
                    current_eps[ind, :] = eps[ind, :]
                    current_U[ind] = proposed_U[ind]
                except:
                    print('Samples were all rejected...skipping')
                    continue
                if i < burn_in and flag_adapt == 1:
                    stepsize = stepsize + hmc_learning_rate * (
                                accept.float().mean() - hmc_opt_accept) * stepsize
                else:
                    samples[cnt * bsz: (cnt + 1) * bsz,
                    :] = current_eps.squeeze()
                    cnt += 1
                logjointHist[:, i] = -current_U.squeeze()
            acceptRate = acceptHist.mean(dim=1)
            return samples, acceptRate, stepsize

        def get_data(args):
            if args.dataset in TOY_DSETS:
                data, labels = skdatasets.make_moons(n_samples=10000, noise=.2)
                data = torch.from_numpy(data).float()
                labels = torch.from_numpy(labels).long()
                dset = TensorDataset(data, labels)
                dload = DataLoader(dset, args.batch_size, True, drop_last=True)
                return dload, dload
            elif args.dataset == "mnist":
                tr_dataset = datasets.MNIST("./data",
                                            transform=transforms.Compose(
                                                [transforms.ToTensor(),
                                                 lambda x: (((
                                                                         255. * x) + torch.rand_like(
                                                     x)) / 256.).view(-1)]),
                                            download=True)
                te_dataset = datasets.MNIST("./data", train=False,
                                            transform=transforms.Compose(
                                                [transforms.ToTensor(),
                                                 lambda x: x.view(-1)]),
                                            download=True)
                tr_dload = DataLoader(tr_dataset, args.batch_size, True,
                                      drop_last=True)
                te_dload = DataLoader(te_dataset, args.batch_size, False)
                return tr_dload, te_dload
            else:
                raise NotImplementedError

        utils.makedirs(args.save_dir)
        if args.dataset in TOY_DSETS:
            logp_net = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(args.data_dim, args.h_dim)),
                nn.LeakyReLU(.2),
                nn.utils.weight_norm(nn.Linear(args.h_dim, args.h_dim)),
                nn.LeakyReLU(.2),
                nn.Linear(args.h_dim, 1, bias=False)
            )
            logp_fn = lambda x: logp_net(x)

            class G(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.generator = nn.Sequential(
                        nn.Linear(args.noise_dim, args.h_dim, bias=False),
                        nn.BatchNorm1d(args.h_dim, affine=True),
                        nn.ReLU(),
                        nn.Linear(args.h_dim, args.h_dim, bias=False),
                        nn.BatchNorm1d(args.h_dim, affine=True),
                        nn.ReLU(),
                        nn.Linear(args.h_dim, args.data_dim)
                    )
                    self.logsigma = nn.Parameter(
                        (torch.zeros(1, ) + 1.).log())
        else:
            logp_net = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(args.data_dim, 1000)),
                nn.LeakyReLU(.2),
                nn.utils.weight_norm(nn.Linear(1000, 500)),
                nn.LeakyReLU(.2),
                nn.utils.weight_norm(nn.Linear(500, 500)),
                nn.LeakyReLU(.2),
                nn.utils.weight_norm(nn.Linear(500, 250)),
                nn.LeakyReLU(.2),
                nn.utils.weight_norm(nn.Linear(250, 250)),
                nn.LeakyReLU(.2),
                nn.utils.weight_norm(nn.Linear(250, 250)),
                nn.LeakyReLU(.2),
                nn.Linear(250, 1, bias=False)
            )
            logp_fn = lambda x: logp_net(x)

            class G(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.generator = nn.Sequential(
                        nn.Linear(args.noise_dim, 500, bias=False),
                        nn.BatchNorm1d(500, affine=True),
                        nn.Softplus(),
                        nn.Linear(500, 500, bias=False),
                        nn.BatchNorm1d(500, affine=True),
                        nn.Softplus(),
                        nn.Linear(500, args.data_dim),
                        nn.Sigmoid()
                    )
                    self.logsigma = nn.Parameter((-torch.ones(1, )))

        g = G()

        e_optimizer = torch.optim.Adam(logp_net.parameters(), lr=args.lr,
                                       betas=[0.5, .999],
                                       weight_decay=args.weight_decay)
        g_optimizer = torch.optim.Adam(g.parameters(), lr=args.lr / 1,
                                       betas=[0.5, .999],
                                       weight_decay=args.weight_decay)

        train_loader, test_loader = get_data(args)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(
            torch.clamp(x, 0, 1), p, normalize=False, nrow=sqrt(x.size(0)))

        def sample_q(n):
            h = torch.randn((n, args.noise_dim)).to(device)
            x_mu = g.generator(h)
            x = x_mu + torch.randn_like(x_mu) * g.logsigma.exp()
            return x, h

        g.train()
        g.to(device)
        logp_net.to(device)

        itr = 0
        stepsize = 1. / args.noise_dim
        for epoch in range(args.n_epochs):
            for x_d, _ in train_loader:
                if args.dataset in TOY_DSETS:
                    x_d = toy_data.inf_train_gen(args.dataset,
                                                 batch_size=args.batch_size)
                    x_d = torch.from_numpy(x_d).float().to(device)
                else:
                    x_d = x_d.to(device)

                # sample from q(x, h)
                x_g, h_g = sample_q(args.batch_size)

                # ebm obj
                ld = logp_fn(x_d)[:, 0]
                lg_detach = logp_fn(x_g.detach())[:, 0]
                logp_obj = (ld - lg_detach).mean()

                # gen obj
                lg = logp_fn(x_g)[:, 0]
                num_samples_posterior = 2
                h_given_x, acceptRate, stepsize = get_samples(
                    g.generator, x_g.detach(), h_g.clone(),
                    g.logsigma.exp().detach(), burn_in=2,
                    num_samples_posterior=num_samples_posterior,
                    leapfrog_steps=5, stepsize=stepsize, flag_adapt=1,
                    hmc_learning_rate=.02, hmc_opt_accept=.67)

                mean_output_summed = torch.zeros_like(x_g)
                mean_output = g.generator(h_given_x)
                # for h in [h_g, h_given_x]:
                for cnt in range(num_samples_posterior):
                    mean_output_summed = mean_output_summed + mean_output[
                                                              cnt * args.batch_size:(
                                                                                                cnt + 1) * args.batch_size]
                mean_output_summed = mean_output_summed / num_samples_posterior

                c = ((
                                 x_g - mean_output_summed) / g.logsigma.exp() ** 2).detach()
                g_error_entropy = torch.mul(c, x_g).mean(0).sum()
                logq_obj = lg.mean() + g_error_entropy

                if itr % 2 == 0:
                    e_loss = -logp_obj + (ld ** 2).mean() * args.p_control
                    e_optimizer.zero_grad()
                    e_loss.backward()
                    e_optimizer.step()
                else:
                    g_loss = -logq_obj
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

                g.logsigma.data.clamp_(np.log(.001), np.log(.2))

                if itr % args.print_every == 0:
                    print(
                        "({}) | log p obj = {:.4f}, log q obj = {:.4f}, sigma = {:.4f} | "
                        "log p(x_d) = {:.4f}, log p(x_m) = {:.4f}, ent = {:.4f} | "
                        "stepsize = {:.4f}".format(
                            itr, logp_obj.item(), logq_obj.item(),
                            g.logsigma.exp().item(),
                            ld.mean().item(), lg.mean().item(),
                            g_error_entropy.item(), stepsize.item()))

                if itr % args.viz_every == 0:
                    if args.dataset in TOY_DSETS:
                        plt.clf()
                        xg = x_g.detach().cpu().numpy()
                        xd = x_d.cpu().numpy()
                        ax = plt.subplot(1, 4, 1, aspect="equal",
                                         title='refined')
                        ax.scatter(xg[:, 0], xg[:, 1], s=1)

                        ax = plt.subplot(1, 4, 2, aspect="equal",
                                         title='data')
                        ax.scatter(xd[:, 0], xd[:, 1], s=1)

                        ax = plt.subplot(1, 4, 3, aspect="equal")
                        logp_net.cpu()
                        utils.plt_flow_density(logp_fn, ax,
                                               low=x_d.min().item(),
                                               high=x_d.max().item())
                        plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                        logp_net.to(device)

                        ax = plt.subplot(1, 4, 4, aspect="equal")
                        logp_net.cpu()
                        utils.plt_flow_density(logp_fn, ax,
                                               low=x_d.min().item(),
                                               high=x_d.max().item(),
                                               exp=False)
                        plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                        logp_net.to(device)
                    else:
                        plot("{}/ref_{}.png".format(args.save_dir, itr),
                             x_g.view(x_g.size(0), *args.data_size))
                        plot("{}/data_{}.png".format(args.save_dir, itr),
                             x_d.view(x_d.size(0), *args.data_size))

                itr += 1
        2/0

    import utils
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    if args.dataset == "mnist":
        args.n_ch = 1
        args.im_sz = 28
    elif args.dataset == "moons":
        args.n_ch = None
        args.im_sz = None
    else:
        args.n_ch = 3
        args.im_sz = 32

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test, dset_train, \
    dset_train_labeled, dload_train_labeled_static, dload_train_vbnorm = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    ref_x = None
    if args.vbnorm:
        ref_x = next(iter(dload_train_vbnorm))[0].to(device)

    sample_q = get_sample_q(args, device)

    momentum_buffer = None
    if args.use_sgld_momentum:
        f, replay_buffer, momentum_buffer = get_model_and_buffer_with_momentum(args, device, sample_q, ref_x)
    else:
        f, replay_buffer = get_model_and_buffer(args, device, sample_q, ref_x)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


    if args.pgan:
        from pgan import G, get_samples
        logp_net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(args.im_sz ** 2, 1000)),
            nn.LeakyReLU(.2),
            nn.utils.weight_norm(nn.Linear(1000, 500)),
            nn.LeakyReLU(.2),
            nn.utils.weight_norm(nn.Linear(500, 500)),
            nn.LeakyReLU(.2),
            nn.utils.weight_norm(nn.Linear(500, 250)),
            nn.LeakyReLU(.2),
            nn.utils.weight_norm(nn.Linear(250, 250)),
            nn.LeakyReLU(.2),
            nn.utils.weight_norm(nn.Linear(250, 250)),
            nn.LeakyReLU(.2),
            nn.Linear(250, 1, bias=False)
        )

        def logp_fn(x):
            if len(x.shape) > 2:
                x = x.reshape(-1, x.shape[-1] ** 2)
            return logp_net(x)

        data_dim = args.im_sz ** 2
        g = G(args.noise_dim, data_dim)

        e_optimizer = t.optim.Adam(logp_net.parameters(), lr=args.lr,
                                       betas=[0.5, .999],
                                       weight_decay=args.weight_decay)
        g_optimizer = t.optim.Adam(g.parameters(), lr=args.lr / 1,
                                       betas=[0.5, .999],
                                       weight_decay=args.weight_decay)

        g.train()
        g.to(device)
        logp_net.to(device)

        def sample_q_pgan(n):
            h = t.randn((n, args.noise_dim)).to(device)
            x_mu = g.generator(h)
            x = x_mu + t.randn_like(x_mu) * g.logsigma.exp()
            return x, h

        pgan_itr = 0
        args.pgan_stepsize = 1. / args.noise_dim

        def pgan_optimize_and_get_sample(pgan_itr, x_p_d):
            x_g, h_g = sample_q_pgan(args.batch_size)

            # ebm obj
            ld = logp_fn(x_p_d)[:, 0]
            lg_detach = logp_fn(x_g.detach())[:, 0]
            logp_obj = (ld - lg_detach).mean()

            # gen obj
            lg = logp_fn(x_g)[:, 0]
            num_samples_posterior = 2
            h_given_x, acceptRate, args.pgan_stepsize = get_samples(
                g.generator, x_g.detach(), h_g.clone(),
                g.logsigma.exp().detach(), burn_in=2,
                num_samples_posterior=num_samples_posterior,
                leapfrog_steps=5, stepsize=args.pgan_stepsize,
                flag_adapt=1,
                hmc_learning_rate=.02, hmc_opt_accept=.67)

            mean_output_summed = t.zeros_like(x_g)
            mean_output = g.generator(h_given_x)
            # for h in [h_g, h_given_x]:
            for cnt in range(num_samples_posterior):
                mean_output_summed = mean_output_summed + mean_output[
                                                          cnt * args.batch_size:(cnt + 1) * args.batch_size]
            mean_output_summed = mean_output_summed / num_samples_posterior

            c = ((x_g - mean_output_summed) / g.logsigma.exp() ** 2).detach()
            g_error_entropy = t.mul(c, x_g).mean(0).sum()
            logq_obj = lg.mean() + g_error_entropy

            if pgan_itr % 2 == 0:
                e_loss = -logp_obj + (
                        ld ** 2).mean() * args.p_control
                e_optimizer.zero_grad()
                e_loss.backward()
                e_optimizer.step()
            else:
                g_loss = -logq_obj
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            g.logsigma.data.clamp_(np.log(args.log_sigma_low),
                                   np.log(args.log_sigma_high))

            if pgan_itr % args.print_every == 0:
                print(
                    "({}) | log p obj = {:.4f}, log q obj = {:.4f}, sigma = {:.4f} | "
                    "log p(x_d) = {:.4f}, log p(x_m) = {:.4f}, ent = {:.4f} | "
                    "stepsize = {:.4f}".format(
                        pgan_itr, logp_obj.item(), logq_obj.item(),
                        g.logsigma.exp().item(),
                        ld.mean().item(), lg.mean().item(),
                        g_error_entropy.item(),
                        args.pgan_stepsize.item()))

            if pgan_itr % args.viz_every == 0:

                plot("{}/ref_{}.png".format(args.save_dir,
                                            pgan_itr),
                     x_g.view(x_g.size(0), *args.data_size))
                plot("{}/data_{}.png".format(args.save_dir,
                                             pgan_itr),
                     x_p_d.view(x_p_d.size(0), *args.data_size))

            x_g = x_g.reshape(x_p_d.shape)

            return x_g


    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    optim_sgld = None
    if args.optim_sgld:
        # TODO other optimizers
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


    if args.pgan:
        while True:
            for x_p_d, _ in dload_train:
                x_p_d = x_p_d.to(device)

                pgan_optimize_and_get_sample(pgan_itr, x_p_d)

                pgan_itr += 1

                if pgan_itr > args.pgan_warmup_iters:
                    break
            if pgan_itr > args.pgan_warmup_iters:
                break

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
                seed_batch = x_p_d.clone() # breaks reference so that we don't change
                # x_p_d at the same time, important for the f = fp - fq calculation
                # to be non-zero

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
                    loss += cond_entropy(logits) * args.ent_min_weight
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

                if args.p_x_weight > 0:  # maximize log p(x)
                    # if args.class_cond_label_prop:
                        # May no longer need class cond samples now
                        # assert args.class_cond_p_x_sample, "need class-conditional samples for psuedo label prop"
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
                        if args.pgan:
                            x_g = pgan_optimize_and_get_sample(pgan_itr, x_p_d)
                            pgan_itr += 1

                            # x_q = x_g.clone().detach()
                            seed_batch = x_g.clone().detach()

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

                        # print(t.sum(t.abs(x_q) - t.abs(x_p_d)))
                        # print(t.sum(t.abs(f(x_q)) - t.abs(f(x_p_d))))
                        # print(fp-fq)

                        l_p_x = -(fp - fq)
                        if cur_iter % args.print_every == 0:
                            print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                           fp - fq))
                        L += args.p_x_weight * l_p_x

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
                        L += cond_entropy(logits) * args.ent_min_weight
                        logits_unlab = f.classify(x_p_d)
                        # Do we need to do the proper dataset weighing? Ie since more
                        # unlabeled than labeled data points
                        # Maybe it's ok
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
                optim.step()

                if args.eval_mode_except_clf:
                    f.eval()

                cur_iter += 1

                if cur_iter % 100 == 0:
                    if args.plot_uncond:
                        if args.class_cond_p_x_sample:
                            assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                            y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                            x_q = sample_q(f, replay_buffer, y=y_q, optim_sgld=optim_sgld,
                                           seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)
                        else:
                            x_q = sample_q(f, replay_buffer, optim_sgld=optim_sgld,
                                           seed_batch=seed_batch, momentum_buffer=momentum_buffer, data=x_p_d)
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
                correct, loss = eval_classification(f, dload_valid, device)
                print("Epoch {}: Valid Loss {}, Valid Acc {}".format(epoch, loss, correct))
                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Best Valid!: {}".format(correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", args, device)
                # test set
                correct, loss = eval_classification(f, dload_test, device)
                print("Epoch {}: Test Loss {}, Test Acc {}".format(epoch, loss, correct))
            f.train()

            if args.dataset == "moons" and correct >= best_valid_acc:
                data,labels= datasets.make_moons(args.n_moons_data, noise=0.1)
                data = t.Tensor(data)
                preds = f.classify(data.to(device))
                preds = preds.argmax(dim=1)
                preds = preds.cpu()
                data1 = data[preds == 0]
                plt.scatter(data1[:,0], data1[:,1], c="orange")
                data2 = data[preds == 1]
                plt.scatter(data2[:,0], data2[:,1], c="blue")

                labeled_pts = dset_train_labeled[:][0]
                labeled_pts_labels = dset_train_labeled[:][1]
                labeled0 = labeled_pts[labeled_pts_labels == 0]
                labeled1 = labeled_pts[labeled_pts_labels == 1]
                # Note labels right now not forced to be class balanced
                # print(sum(labeled_pts_labels))
                plt.scatter(labeled0[:,0], labeled0[:,1], c="green")
                plt.scatter(labeled1[:,0], labeled1[:,1], c="red")
                print("Saving figure")
                plt.savefig("moonsvis.png")
                # plt.show()

        checkpoint(f, replay_buffer, "last_ckpt.pt", args, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="moons", choices=["cifar10", "svhn", "mnist", "cifar100", "moons"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    #labels was -1?
    # parser.add_argument("--labels_per_class", type=int, default=-1,
    #                     help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--labels_per_class", type=int, default=10,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    # parser.add_argument("--batch_size", type=int, default=64)
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
    # parser.add_argument("--n_valid", type=int, default=50)
    parser.add_argument("--semi-supervised", type=bool, default=False)
    # parser.add_argument("--vat", type=bool, default=False)
    parser.add_argument("--vat", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--vat_weight", type=float, default=1.0)
    parser.add_argument("--n_moons_data", type=float, default=500)
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
    parser.add_argument("--mnist_no_logit_transform", action="store_true", help="Run MNIST without logit transform")
    parser.add_argument("--mnist_no_crop", action="store_true", help="Run MNIST without crop")
    parser.add_argument("--score_match", action="store_true", help="Note: so far implemented only for p(x). Use score matching instead of SGLD in training JEM")
    parser.add_argument("--swish", action="store_true", help="Use swish activation on NN instead of ReLU")
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
    parser.add_argument("--psgld", action="store_true", help="Use Preconditioned SGLD")
    parser.add_argument("--psgld_alpha", type=float, default=0.99)
    parser.add_argument("--psgld_lambda", type=float, default=1e-1)
    parser.add_argument("--psgld_div_mean", action="store_true")
    parser.add_argument("--optim_sgld", action="store_true", help="Use SGLD Optimizer")
    parser.add_argument("--optim_sgld_momentum", type=float, default=0.0)
    parser.add_argument("--use_cd", action="store_true", help="Use contrastive divergence instead of persistent contrastive divergence (initialize from data instead of saved replay buffer/previous samples")
    parser.add_argument("--svhn_logit_transform", action="store_true", help="Run SVHN with logit transform")
    parser.add_argument("--use_sgld_momentum", action="store_true")
    parser.add_argument("--sgld_momentum", type=float, default=0.9)
    parser.add_argument("--cnn_avg_pool_kernel", type=int, default=6)
    parser.add_argument("--use_nn", action="store_true", help="Use NN (4 layer MLP)")
    parser.add_argument("--buffer_reinit_from_data", action="store_true", help="For PCD replay buffer, reinitialize from data points rather than random points")
    parser.add_argument("--pgan", action="store_true", help="Use PGAN to generate samples")
    parser.add_argument("--h_dim", type=int, default=100)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--p_control", type=float, default=0.0)
    parser.add_argument("--log_sigma_low", type=float, default=.001)
    parser.add_argument("--log_sigma_high", type=float, default=0.2)
    parser.add_argument("--viz_every", type=int, default=100, help="Iterations between visualization of reference for PGAN")
    parser.add_argument("--pgan_warmup_iters", type=int, default=80000,
                        help="number of iters to train PGAN before training EBM")
    parser.add_argument("--pgan_test", action="store_true", help="TESTING ONLY")


    args = parser.parse_args()
    if args.dataset == "cifar100":
        args.n_classes = 100
    elif args.dataset == "moons":
        args.n_classes = 2
    else:
        args.n_classes = 10
    if args.vat:
        print("Running VAT")


    main(args)
