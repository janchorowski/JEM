import argparse
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
import hmc
from singular import find_extreme_singular_vectors, log_sigular_values_sum_bound
TOY_DSETS = ("moons", "circles", "8gaussians", "pinwheel", "2spirals", "checkerboard", "rings", "swissroll")

import torch.nn.init as nninit

def avg_pool2d(x):
    '''Twice differentiable implementation of 2x2 average pooling.'''
    return (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4


def brute_force_jac(x_g, h_g):
    jac = torch.zeros((x_g.size(0), x_g.size(1), h_g.size(1)))
    for d in range(x_g.size(1)):
        j = torch.autograd.grad(x_g[:, d].sum(), h_g, retain_graph=True)[0]
        jac[:, d, :] = j
    return jac

def condition_number(j):
    cvals = []
    hvals = []
    lvals = []
    for i in range(j.size(0)):
        u, s, v = torch.svd(j[i], compute_uv=False)
        cvals.append((s[0] / s[-1])[None])
        hvals.append((s[0])[None])
        lvals.append((s[-1])[None])
    return torch.cat(cvals), torch.cat(hvals), torch.cat(lvals)


class GeneratorBlock(nn.Module):
    '''ResNet-style block for the generator model.'''

    def __init__(self, in_chans, out_chans, upsample=False):
        super().__init__()

        self.upsample = upsample

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.upsample:
            shortcut = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=False)
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.bn2(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)

        return x + shortcut

class ResNetGenerator(nn.Module):
    '''The generator model.'''

    def __init__(self):
        super().__init__()

        feats = 128
        self.input_linear = nn.Linear(128, 4*4*feats)
        self.block1 = GeneratorBlock(feats, feats, upsample=True)
        self.block2 = GeneratorBlock(feats, feats, upsample=True)
        self.block3 = GeneratorBlock(feats, feats, upsample=True)
        self.output_bn = nn.BatchNorm2d(feats)
        self.output_conv = nn.Conv2d(feats, 3, kernel_size=3, padding=1)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.input_linear else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

        self.last_output = None

    def forward(self, *inputs):
        x = inputs[0]

        x = self.input_linear(x)
        x = x.view(-1, 128, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_bn(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.output_conv(x)
        x = nn.functional.tanh(x)

        self.last_output = x

        return x

class DiscriminatorBlock(nn.Module):
    '''ResNet-style block for the discriminator model.'''

    def __init__(self, in_chans, out_chans, downsample=False, first=False):
        super().__init__()

        self.downsample = downsample
        self.first = first

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, *inputs):
        x = inputs[0]

        if self.downsample:
            shortcut = avg_pool2d(x)
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        if not self.first:
            x = nn.functional.relu(x, inplace=False)
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.conv2(x)
        if self.downsample:
            x = avg_pool2d(x)

        return x + shortcut

class ResNetDiscriminator(nn.Module):
    '''The discriminator (aka critic) model.'''

    def __init__(self):
        super().__init__()

        feats = 128
        self.block1 = DiscriminatorBlock(3, feats, downsample=True, first=True)
        self.block2 = DiscriminatorBlock(feats, feats, downsample=True)
        self.block3 = DiscriminatorBlock(feats, feats, downsample=False)
        self.block4 = DiscriminatorBlock(feats, feats, downsample=False)
        self.output_linear = nn.Linear(128, 1)

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain('relu')
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.block1.conv1 else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

    def forward(self, *inputs):
        x = inputs[0]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = nn.functional.relu(x, inplace=False)
        x = x.mean(-1, keepdim=False).mean(-1, keepdim=False)
        x = x.view(-1, 128)
        x = self.output_linear(x)

        return x

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)

def update_logp(u, u_mu, std):
    return distributions.Normal(u_mu, std).log_prob(u).flatten(start_dim=1).sum(1)

def MALA(vars, logp_fn, step_lr):
    step_std = (2 * step_lr) ** .5
    logp_vars = logp_fn(*vars)
    grads = torch.autograd.grad(logp_vars.sum(), vars)
    updates_mu = [v + step_lr * g for v, g in zip(vars, grads)]
    updates = [u_mu + step_std * torch.randn_like(u_mu) for u_mu in updates_mu]
    logp_updates = logp_fn(*updates)
    reverse_grads = torch.autograd.grad(logp_updates.sum(), updates)
    reverse_updates_mu = [v + step_lr * g for v, g in zip(updates, reverse_grads)]

    logp_forward = sum([update_logp(u, u_mu, step_std) for u, u_mu in zip(updates, updates_mu)])
    logp_backward = sum([update_logp(v, ru_mu, step_std) for v, ru_mu in zip(vars, reverse_updates_mu)])

    logp_accept = logp_updates + logp_backward - logp_vars - logp_forward
    p_accept = logp_accept.exp()
    accept = (torch.rand_like(p_accept) < p_accept).float()
    if args.dataset in ("svhn", "cifar10"):
        #next_vars = [accept[:, None, None, None] * u_v + (1 - accept[:, None, None, None]) * v for u_v, v in zip(updates, vars)]
        next_vars = []
        for u_v, v in zip(updates, vars):
            if len(u_v.size()) == 4:
                next_vars.append(accept[:, None, None, None] * u_v + (1 - accept[:, None, None, None]) * v)
            else:
                next_vars.append(accept[:, None] * u_v + (1 - accept[:, None]) * v)
    else:
        next_vars = [accept[:, None] * u_v + (1 - accept[:, None]) * v for u_v, v in zip(updates, vars)]
    return next_vars, accept.mean()


def main(args):
    utils.makedirs(args.save_dir)
    cn_sgld_dir = "{}/{}".format(args.save_dir, "condition_number")
    utils.makedirs(cn_sgld_dir)
    h_sgld_dir = "{}/{}".format(args.save_dir, "largest_sv")
    utils.makedirs(h_sgld_dir)
    l_sgld_dir = "{}/{}".format(args.save_dir, "smallest_sv")
    utils.makedirs(l_sgld_dir)
    data_sgld_dir = "{}/{}".format(args.save_dir, "data_sgld")
    utils.makedirs(data_sgld_dir)
    gen_sgld_dir = "{}/{}".format(args.save_dir, "generator_sgld")
    utils.makedirs(gen_sgld_dir)
    z_sgld_dir = "{}/{}".format(args.save_dir, "z_only_sgld")
    utils.makedirs(z_sgld_dir)

    data_sgld_chain_dir = "{}/{}_chain".format(args.save_dir, "data_sgld_chain")
    utils.makedirs(data_sgld_chain_dir)
    gen_sgld_chain_dir = "{}/{}_chain".format(args.save_dir, "generator_sgld_chain")
    utils.makedirs(gen_sgld_chain_dir)
    z_sgld_chain_dir = "{}/{}_chain".format(args.save_dir, "z_only_sgld_chain")
    utils.makedirs(z_sgld_chain_dir)
    logp_net, g = get_models(args)

    e_optimizer = torch.optim.Adam(logp_net.parameters(), lr=args.lr, betas=[.0, .9], weight_decay=args.weight_decay)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=args.lr / 1, betas=[.0, .9], weight_decay=args.weight_decay)

    train_loader, test_loader, plot = get_data(args)

    def sample_q(n, requires_grad=False):
        h = torch.randn((n, args.noise_dim)).to(device)
        if requires_grad:
            h.requires_grad_()
        x_mu = g.generator(h)
        x = x_mu + torch.randn_like(x_mu) * g.logsigma.exp()
        return x, h

    def logq_joint(x, h):
        logph = distributions.Normal(0, 1).log_prob(h).sum(1)
        px_given_h = distributions.Normal(g.generator(h), g.logsigma.exp())
        logpx_given_h = px_given_h.log_prob(x).flatten(start_dim=1).sum(1)
        return logpx_given_h + logph

    g.train()
    g.to(device)
    logp_net.train()
    logp_net.to(device)

    itr = 0
    stepsize = 1. / args.noise_dim
    sgld_lr = 1. / args.noise_dim
    sgld_lr_z = 1. / args.noise_dim
    sgld_lr_zne = 1. / args.noise_dim
    for epoch in range(args.n_epochs):
        for x_d, _ in train_loader:
            if args.dataset in TOY_DSETS:
                x_d = toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)
                x_d = torch.from_numpy(x_d).float().to(device)
            else:
                x_d = x_d.to(device)

            # sample from q(x, h)
            x_g, h_g = sample_q(args.batch_size)
            x_g_ref = x_g
            # ebm obj
            ld = logp_net(x_d).squeeze()
            lg_detach = logp_net(x_g_ref.detach()).squeeze()
            logp_obj = (ld - lg_detach).mean()

            e_loss = -logp_obj + (ld ** 2).mean() * args.p_control
            if itr % args.e_iters == 0:
                e_optimizer.zero_grad()
                e_loss.backward()
                e_optimizer.step()

            # gen obj
            lg = logp_net(x_g).squeeze()

            if args.gp == 0:
                if args.my_single_sample:
                    logq = logq_joint(x_g.detach(), h_g.detach())
                    # mine
                    logq_obj = lg.mean() - args.ent_weight * logq.mean()
                    g_error_entropy = -logq.mean()
                elif args.adji_single_sample:
                    # adji
                    mean_output_summed = g.generator(h_g)
                    c = ((x_g - mean_output_summed) / g.logsigma.exp() ** 2).detach()
                    g_error_entropy = torch.mul(c, x_g).mean(0).sum()
                    logq_obj = lg.mean() + args.ent_weight * g_error_entropy
                elif args.sv_bound:
                    v, t = find_extreme_singular_vectors(g.generator, h_g, args.niters, args.v_norm)
                    log_sv = log_sigular_values_sum_bound(g.generator, h_g, v, args.v_norm)
                    logpx = - log_sv.mean() - distributions.Normal(0, g.logsigma.exp()).entropy() * args.data_dim
                    g_error_entropy = logpx
                    logq_obj = lg.mean() - args.ent_weight * logpx
                else:
                    num_samples_posterior = 2
                    h_given_x, acceptRate, stepsize = hmc.get_gen_posterior_samples(
                        g.generator, x_g.detach(), h_g.clone(), g.logsigma.exp().detach(), burn_in=2,
                        num_samples_posterior=num_samples_posterior, leapfrog_steps=5, stepsize=stepsize, flag_adapt=1,
                        hmc_learning_rate=.02, hmc_opt_accept=.67)

                    mean_output_summed = torch.zeros_like(x_g)
                    mean_output = g.generator(h_given_x)
                    # for h in [h_g, h_given_x]:
                    for cnt in range(num_samples_posterior):
                        mean_output_summed = mean_output_summed + mean_output[cnt*args.batch_size:(cnt+1)*args.batch_size]
                    mean_output_summed = mean_output_summed / num_samples_posterior

                    c = ((x_g - mean_output_summed) / g.logsigma.exp() ** 2).detach()
                    g_error_entropy = torch.mul(c, x_g).mean(0).sum()
                    logq_obj = lg.mean() + args.ent_weight * g_error_entropy
            else:
                x_g, h_g = sample_q(args.batch_size, requires_grad=True)
                if args.brute_force:
                    jac = torch.zeros((x_g.size(0), x_g.size(1), h_g.size(1)))
                    j = torch.autograd.grad(x_g[:, 0].sum(), h_g, retain_graph=True)[0]
                    jac[:, 0, :] = j
                    j = torch.autograd.grad(x_g[:, 1].sum(), h_g, retain_graph=True)[0]
                    jac[:, 1, :] = j
                    u, s, v = torch.svd(jac)
                    logs = s.log()
                    logpx = 0 - logs.sum(1)
                    g_error_entropy = logpx.mean()
                    logq_obj = lg.mean() - args.ent_weight * logpx.mean()

                else:
                    eps = torch.randn_like(x_g)
                    epsJ = torch.autograd.grad(x_g, h_g, grad_outputs=eps, retain_graph=True)[0]
                    #eps2 = torch.randn_like(x_g)
                    #epsJ2 = torch.autograd.grad(x_g, h_g, grad_outputs=eps2, retain_graph=True)[0]
                    epsJtJeps = (epsJ * epsJ).sum(1)
                    g_error_entropy = ((epsJtJeps - args.gp) ** 2).mean()
                    logq_obj = lg.mean() - args.ent_weight * g_error_entropy


            g_loss = -logq_obj
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if args.clamp:
                g.logsigma.data.clamp_(np.log(.01), np.log(.0101))
            else:
                g.logsigma.data.clamp_(np.log(.01), np.log(.3))

            if itr % args.print_every == 0:
                print("({}) | log p obj = {:.4f}, log q obj = {:.4f}, sigma = {:.4f} | "
                      "log p(x_d) = {:.4f}, log p(x_m) = {:.4f}, ent = {:.4f} | "
                      "sgld_lr = {}, sgld_lr_z = {}, sgld_lr_zne = {} | stepsize = {}".format(
                    itr, logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(),
                    ld.mean().item(), lg_detach.mean().item(), g_error_entropy.item(),
                    sgld_lr, sgld_lr_z, sgld_lr_zne, stepsize))

            if itr % args.viz_every == 0:
                if args.dataset in TOY_DSETS:
                    plt.clf()
                    xg = x_g_ref.detach().cpu().numpy()
                    xd = x_d.cpu().numpy()

                    ax = plt.subplot(1, 4, 1, aspect="equal", title='refined')
                    ax.scatter(xg[:, 0], xg[:, 1], s=1)

                    ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
                    ax.scatter(xd[:, 0], xd[:, 1], s=1)

                    ax = plt.subplot(1, 4, 3, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_net, ax, low=x_d.min().item(), high=x_d.max().item())
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)

                    ax = plt.subplot(1, 4, 4, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_net, ax, low=x_d.min().item(), high=x_d.max().item(), exp=False)
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)

                    x_g, h_g = sample_q(args.batch_size, requires_grad=True)
                    jac = torch.zeros((x_g.size(0), x_g.size(1), h_g.size(1)))


                    j = torch.autograd.grad(x_g[:, 0].sum(), h_g, retain_graph=True)[0]
                    jac[:, 0, :] = j
                    j = torch.autograd.grad(x_g[:, 1].sum(), h_g, retain_graph=True)[0]
                    jac[:, 1, :] = j
                    u, s, v = torch.svd(jac)

                    s1, s2 = s[:, 0].detach(), s[:, 1].detach()
                    plt.clf()
                    plt.hist(s1.numpy(), alpha=.75)
                    plt.hist(s2.numpy(), alpha=.75)
                    plt.savefig("{}/{}_svd.png".format(args.save_dir, itr))

                    plt.clf()
                    plt.hist(s1.log().numpy(), alpha=.75)
                    plt.hist(s2.log().numpy(), alpha=.75)
                    plt.savefig("{}/{}_log_svd.png".format(args.save_dir, itr))
                else:
                    x_g, h_g = sample_q(args.batch_size, requires_grad=True)
                    J = brute_force_jac(x_g, h_g)
                    c, h, l = condition_number(J)
                    plt.clf()
                    plt.hist(c.numpy())
                    plt.savefig("{}/cn_{}.png".format(cn_sgld_dir, itr))
                    plt.clf()
                    plt.hist(h.numpy())
                    plt.savefig("{}/large_s_{}.png".format(h_sgld_dir, itr))
                    plt.clf()
                    plt.hist(l.numpy())
                    plt.savefig("{}/small_s_{}.png".format(l_sgld_dir, itr))
                    plt.clf()


                    plot("{}/{}_init.png".format(data_sgld_dir, itr),
                         x_g.view(x_g.size(0), *args.data_size))
                    #plot("{}/{}_ref.png".format(args.save_dir, itr), x_g_ref.view(x_g.size(0), *args.data_size))
                    # input space sgld
                    x_sgld = x_g.clone()
                    steps = [x_sgld.clone()]
                    accepts = []
                    for k in range(args.sgld_steps):
                        [x_sgld], a = MALA([x_sgld], lambda x: logp_net(x).squeeze(), sgld_lr)
                        steps.append(x_sgld.clone())
                        accepts.append(a.item())
                    ar = np.mean(accepts)
                    print("accept rate: {}".format(ar))
                    sgld_lr = sgld_lr + .2 * (ar - .57) * sgld_lr
                    plot("{}/{}_ref.png".format(data_sgld_dir, itr),
                         x_sgld.view(x_g.size(0), *args.data_size))

                    chain = torch.cat([step[0][None] for step in steps], 0)
                    plot("{}/{}.png".format(data_sgld_chain_dir, itr),
                         chain.view(chain.size(0), *args.data_size))

                    # latent space sgld
                    eps_sgld = torch.randn_like(x_g)
                    z_sgld = torch.randn((eps_sgld.size(0), args.noise_dim)).to(eps_sgld.device)
                    vs = (z_sgld.requires_grad_(), eps_sgld.requires_grad_())
                    steps = [vs]
                    accepts = []
                    gfn = lambda z, e: g.generator(z) + g.logsigma.exp() * e
                    efn = lambda z, e: logp_net(gfn(z, e)).squeeze()
                    x_sgld = gfn(z_sgld, eps_sgld)
                    plot("{}/{}_init.png".format(gen_sgld_dir, itr), x_sgld.view(x_g.size(0), *args.data_size))
                    for k in range(args.sgld_steps):
                        vs, a = MALA(vs, efn, sgld_lr_z)
                        steps.append([v.clone() for v in vs])
                        accepts.append(a.item())
                    ar = np.mean(accepts)
                    print("accept rate: {}".format(ar))
                    sgld_lr_z = sgld_lr_z + .2 * (ar - .57) * sgld_lr_z
                    z_sgld, eps_sgld = steps[-1]
                    x_sgld = gfn(z_sgld, eps_sgld)
                    plot("{}/{}_ref.png".format(gen_sgld_dir, itr), x_sgld.view(x_g.size(0), *args.data_size))

                    z_steps, eps_steps = zip(*steps)
                    z_chain = torch.cat([step[0][None] for step in z_steps], 0)
                    eps_chain = torch.cat([step[0][None] for step in eps_steps], 0)
                    chain = gfn(z_chain, eps_chain)
                    plot("{}/{}.png".format(gen_sgld_chain_dir, itr),
                         chain.view(chain.size(0), *args.data_size))

                    # latent space sgld no eps
                    z_sgld = torch.randn((eps_sgld.size(0), args.noise_dim)).to(eps_sgld.device)
                    vs = (z_sgld.requires_grad_(),)
                    steps = [vs]
                    accepts = []
                    gfn = lambda z: g.generator(z)
                    efn = lambda z: logp_net(gfn(z)).squeeze()
                    x_sgld = gfn(z_sgld)
                    plot("{}/{}_init.png".format(z_sgld_dir, itr), x_sgld.view(x_g.size(0), *args.data_size))
                    for k in range(args.sgld_steps):
                        vs, a = MALA(vs, efn, sgld_lr_zne)
                        steps.append([v.clone() for v in vs])
                        accepts.append(a.item())
                    ar = np.mean(accepts)
                    print("accept rate: {}".format(ar))
                    sgld_lr_zne = sgld_lr_zne + .2 * (ar - .57) * sgld_lr_zne
                    z_sgld, = steps[-1]
                    x_sgld = gfn(z_sgld)
                    plot("{}/{}_ref.png".format(z_sgld_dir, itr), x_sgld.view(x_g.size(0), *args.data_size))

                    z_steps = [s[0] for s in steps]
                    z_chain = torch.cat([step[0][None] for step in z_steps], 0)
                    chain = gfn(z_chain)
                    plot("{}/{}.png".format(z_sgld_chain_dir, itr),
                         chain.view(chain.size(0), *args.data_size))


            itr += 1



def get_data(args):
    if args.dataset in TOY_DSETS:
        data, labels = skdatasets.make_moons(n_samples=10000, noise=.2)
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        dset = TensorDataset(data, labels)
        dload = DataLoader(dset, args.batch_size, True, drop_last=True)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x, p, normalize=False, nrow=sqrt(x.size(0)))
        return dload, dload, plot
    elif args.dataset == "mnist":
        tr_dataset = datasets.MNIST("./data",
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  lambda x: (((255. * x) + torch.rand_like(x)) / 256.).view(-1)]),
                                    download=True)
        te_dataset = datasets.MNIST("./data", train=False,
                                    transform=transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)]),
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)

        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(x, p, normalize=False, nrow=sqrt(x.size(0)))

        return tr_dload, te_dload, plot
    elif args.dataset == "svhn":
        tr_dataset = datasets.SVHN("./data",
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  lambda x: (
                                                                  ((255. * x) + torch.rand_like(x)) / 256.),
                                                                  lambda x: 2 * x - 1]),
                                    download=True)
        te_dataset = datasets.SVHN("./data", split='test',
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  lambda x: 2 * x - 1]),
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(.5 * x + .5, p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    elif args.dataset == "cifar10":
        tr_dataset = datasets.CIFAR10("./data",
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  lambda x: (
                                                                  ((255. * x) + torch.rand_like(x)) / 256.),
                                                                  lambda x: 2 * x - 1]),
                                    download=True)
        te_dataset = datasets.CIFAR10("./data", train=False,
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  lambda x: 2 * x - 1]),
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
        plot = lambda p, x: torchvision.utils.save_image(.5 * x + .5, p, normalize=False, nrow=sqrt(x.size(0)))
        return tr_dload, te_dload, plot

    else:
        raise NotImplementedError


def get_models(args):
    if args.dataset in TOY_DSETS:
        logp_net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(args.data_dim, args.h_dim)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(args.h_dim, args.h_dim)),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(args.h_dim, 1, bias=False)
        )

        class G(nn.Module):
            def __init__(self):
                super().__init__()
                self.generator = nn.Sequential(
                    nn.Linear(args.noise_dim, args.h_dim, bias=False),
                    nn.BatchNorm1d(args.h_dim, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.h_dim, args.h_dim, bias=False),
                    nn.BatchNorm1d(args.h_dim, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.h_dim, args.data_dim, bias=False)
                )
                self.logsigma = nn.Parameter((torch.zeros(1, ) + .01))
                self.post_logsigma = nn.Parameter(torch.zeros(args.noise_dim,))
    elif args.dataset == "mnist":
        logp_net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(args.data_dim, 1000)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(1000, 500)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(500, 500)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(500, 250)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(250, 250)),
            nn.LeakyReLU(.2, inplace=True),
            nn.utils.weight_norm(nn.Linear(250, 250)),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(250, 1, bias=False)
        )

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
                    nn.Linear(500, args.data_dim, bias=False),
                    nn.Sigmoid()
                )
                self.logsigma = nn.Parameter((torch.ones(1, ) * .01).log())
                self.post_logsigma = nn.Parameter(torch.zeros(args.noise_dim, ))

    elif args.dataset == "svhn" or args.dataset == "cifar10":
        # logp_net = nn.Sequential(
        #     nn.utils.weight_norm(nn.Conv2d(args.data_size[0], 64, 3, 1, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.utils.weight_norm(nn.Conv2d(64, 64, 3, 1, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.utils.weight_norm(nn.Conv2d(64, 64, 3, 2, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.Dropout2d(.5),
        #     nn.utils.weight_norm(nn.Conv2d(64, 128, 3, 1, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.utils.weight_norm(nn.Conv2d(128, 128, 3, 1, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.utils.weight_norm(nn.Conv2d(128, 128, 3, 2, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.Dropout2d(.5),
        #     nn.utils.weight_norm(nn.Conv2d(128, 128, 3, 1, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.utils.weight_norm(nn.Conv2d(128, 128, 2, 1, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.utils.weight_norm(nn.Conv2d(128, 128, 1, 1, 0)),
        #     nn.LeakyReLU(.2, inplace=True),
        #     nn.utils.weight_norm(nn.Conv2d(128, 1, 1, 1, 0)),
        # )
        if not args.resnet:
            logp_net = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 16 x 16
                nn.Conv2d(64, 128, 4, 2, 1),
                #nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 8 x 8
                nn.Conv2d(128, 256, 4, 2, 1),
                #nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 4 x 4
                nn.Conv2d(256, 512, 4, 2, 1),
                #nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 2 x 2
                nn.Conv2d(512, 1, 2, 1, 0, bias=False),
                #nn.Sigmoid()
            )
        else:
            logp_net = ResNetDiscriminator()



        class Generator(nn.Module):
            def __init__(self):
                super().__init__()
                # self.first = nn.Sequential(
                #     nn.ConvTranspose2d(args.noise_dim, 512, 4, 1, 0, bias=False),
                #     nn.BatchNorm2d(512),
                #     nn.ReLU(inplace=True),
                #     nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False),
                #     nn.BatchNorm2d(256),
                #     nn.ReLU(inplace=True),
                #     nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False),
                #     nn.BatchNorm2d(128),
                #     nn.ReLU(inplace=True),
                #     nn.ConvTranspose2d(128, 3, 5, 2, 2, 1),
                #     nn.Tanh()
                # )
                ngf = 64
                self.first = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(args.noise_dim, ngf * 8, 2, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
                    nn.Tanh()
                    # state size. (nc) x 64 x 64
                )

            def forward(self, x):
                x = x.view(x.size(0), -1, 1, 1)
                x = self.first(x)
                return x

        class G(nn.Module):
            def __init__(self):
                super().__init__()
                self.generator = ResNetGenerator() if args.resnet else Generator()
                self.logsigma = nn.Parameter((torch.ones(1, ) * .01).log())
                self.post_logsigma = nn.Parameter(torch.zeros(args.noise_dim, ))

    return logp_net, G()




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="circles")#, choices=["mnist", "moons", "circles"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--v_norm", type=float, default=.01)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--labels_per_class", type=int, default=10,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--h_dim", type=int, default=100)
    parser.add_argument("--sgld_steps", type=int, default=100)
    parser.add_argument("--noise_dim", type=int, default=2)
    parser.add_argument("--e_iters", type=int, default=1)
    parser.add_argument("--g_iters", type=int, default=1)
    parser.add_argument("--niters", type=int, default=10)
    # loss weighting
    parser.add_argument("--ent_weight", type=float, default=1.)
    parser.add_argument("--gp", type=float, default=0.)
    # regularization
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--p_control", type=float, default=0.0)
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=10,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--sgld_step", type=float, default=.01)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='/tmp/pgan_simp')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=20, help="Iterations between print")
    parser.add_argument("--viz_every", type=int, default=2000, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--semi-supervised", type=bool, default=False)
    parser.add_argument("--vat", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--stagger", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--resnet", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--hmc", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--refine", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--refine_latent", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--brute_force", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--my_single_sample", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--adji_single_sample", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--clamp", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--sv_bound", action="store_true", help="Run VAT instead of JEM")

    args = parser.parse_args()
    if args.dataset in TOY_DSETS:
        args.data_dim = 2
    elif args.dataset == "mnist":
        args.data_dim = 784
        args.data_size = (1, 28, 28)
    elif args.dataset == "svhn" or args.dataset == "cifar10":
        args.data_dim = 32 * 32 * 3
        args.data_size = (3, 32, 32)

    main(args)
