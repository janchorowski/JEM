import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.distributions as distributions
import torch.utils
import torch.utils.data
import hmc
import pandas as pd
import utils
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is {}".format(device))
stepsize = 1.


class EBM(nn.Module):
    def __init__(self, net, yind, init_dist, base_dist,
                 sgld_lr, sgld_std, mcmc_steps, sample_init,
                 p_control, n_control, buffer_size, reinit_freq, sampler="sgld"):
        super().__init__()
        self.net = net
        self.base_dist = base_dist
        self.sample_init = sample_init
        self.init_dist = init_dist
        self.yind = yind
        self.sgld_std = sgld_std
        self.sgld_lr = sgld_lr if sgld_lr > 0 else sgld_std ** 2 / 2
        self.mcmc_steps = mcmc_steps
        self.p_control = p_control
        self.n_control = n_control
        self.buffer = None#init_dist.sample((buffer_size,))
        self.reinit_freq = reinit_freq
        self.stepsize = 1. / init_dist.sample().size(0)
        self.ar = 0.
        self.sampler = sampler


    def forward(self, x):
        out = self.net(x)
        if self.base_dist:
            out = out - (x * x).sum(1, keepdim=True) / 2.
        return out

    def logpx_objective(self, x):

        y_init = self.init_dist.sample((x.size(0),))[:, self.yind][:, None].to(x.device)

        log_fn = lambda y: self(torch.cat([x, y], dim=1)).squeeze()

        if self.sampler == "sgld":
            y_fake = hmc.sgld_sample(log_fn, y_init, l=self.sgld_lr, e=self.sgld_std, n_steps=self.mcmc_steps)
        else:
            y_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, y_init, args.mcmc_steps, 1, 5,
                                                       self.stepsize, 1, .02, .67)
            self.stepsize = stepsize
            self.ar = ar.mean().item()

        xy = torch.cat([x, y_fake], dim=1)
        logp_real = self(xy)

        # initialize
        if self.sample_init == "cd":
            xy_init = xy.clone().detach()
        elif self.sample_init == "pcd":
            if self.buffer is None or self.buffer.size(0) < args.buffer_size:
                xy_init = xy.clone().detach()
            else:
                inds = list(range(self.buffer.size(0)))
                inds = np.random.choice(inds, x.size(0), replace=False)
                inds = torch.from_numpy(inds)
                xy_init = self.buffer[inds].to(device)
        else:
            xy_init = self.init_dist.sample((x.size(0),)).to(x.device)
        log_fn = lambda xy: self(xy).squeeze()

        if self.sampler == "sgld":
            xy_fake = hmc.sgld_sample(log_fn, xy_init, l=self.sgld_lr, e=self.sgld_std, n_steps=self.mcmc_steps)
        else:
            xy_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, xy_init, args.mcmc_steps, 1, 5,
                                                        self.stepsize, 1, .02, .67)
            self.stepsize = stepsize
            self.ar = ar.mean().item()

        logp_fake = self(xy_fake)

        obj = logp_real.mean() - logp_fake.mean()
        loss = -obj + (logp_real ** 2).mean() * self.p_control + (logp_fake ** 2).mean() * self.n_control
        if self.sample_init == "pcd":
            if self.buffer is None or self.buffer.size(0) < args.buffer_size:
                if self.buffer is None:
                    self.buffer = xy_fake.cpu().detach()
                else:
                    self.buffer = torch.cat([self.buffer, xy_fake.cpu().detach()], 0)
                print("Filling buffer {}".format(self.buffer.size(0)))
            else:
                u = torch.rand((x.size(0)))
                reinit = (u < self.reinit_freq).float()[:, None]
                new_samples = self.init_dist.sample((x.size(0),))
                updated_samples = new_samples * reinit + xy_fake.cpu().detach() * (1 - reinit)
                self.buffer[inds] = updated_samples
        return {'loss': loss, 'real': logp_real.mean().item(), 'fake': logp_fake.mean().item()}

    def logpxy_objective(self, x, y):
        xy = torch.cat([x, y[:, None]], dim=1)
        logp_real = self(xy)

        # initialize
        if self.sample_init == "cd":
            xy_init = xy.clone().detach()
        elif self.sample_init == "pcd":
            if self.buffer is None or self.buffer.size(0) < args.buffer_size:
                xy_init = xy.clone().detach()
            else:
                inds = list(range(self.buffer.size(0)))
                inds = np.random.choice(inds, x.size(0), replace=False)
                inds = torch.from_numpy(inds)
                xy_init = self.buffer[inds].to(device)
        else:
            xy_init = self.init_dist.sample((x.size(0),)).to(x.device)
        log_fn = lambda xy: self(xy).squeeze()

        if self.sampler == "sgld":
            xy_fake = hmc.sgld_sample(log_fn, xy_init, l=self.sgld_lr, e=self.sgld_std, n_steps=self.mcmc_steps)
        else:
            xy_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, xy_init, args.mcmc_steps, 1, 5,
                                                        self.stepsize, 1, .02, .67)
            self.stepsize = stepsize
            self.ar = ar.mean().item()

        logp_fake = self(xy_fake)

        obj = logp_real.mean() - logp_fake.mean()
        loss = -obj + (logp_real ** 2).mean() * self.p_control + (logp_fake ** 2).mean() * self.n_control
        if self.sample_init == "pcd":
            if self.buffer is None or self.buffer.size(0) < args.buffer_size:
                if self.buffer is None:
                    self.buffer = xy_fake.cpu().detach()
                else:
                    self.buffer = torch.cat([self.buffer, xy_fake.cpu().detach()], 0)
                print("Filling buffer {}".format(self.buffer.size(0)))
            else:
                u = torch.rand((x.size(0)))
                reinit = (u < self.reinit_freq).float()[:, None]
                new_samples = self.init_dist.sample((x.size(0),))
                updated_samples = new_samples * reinit + xy_fake.cpu().detach() * (1 - reinit)
                self.buffer[inds] = updated_samples
        return {'loss': loss, 'real': logp_real.mean().item(), 'fake': logp_fake.mean().item()}

    def logpy_given_x_objective(self, x, y):
        xy = torch.cat([x, y[:, None]], dim=1)
        logp_real = self(xy)

        if self.sample_init == "cd":
            y_init = y.detach()[:, None]
        elif self.sample_init == "pcd":
            raise ValueError
        else:
            y_init = self.init_dist.sample((x.size(0),))[:, self.yind][:, None].to(x.device)

        log_fn = lambda y: self(torch.cat([x, y], dim=1)).squeeze()

        if self.sampler == "sgld":
            y_fake = hmc.sgld_sample(log_fn, y_init, l=self.sgld_lr, e=self.sgld_std, n_steps=self.mcmc_steps)
        else:
            y_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, y_init, args.mcmc_steps, 1, 5,
                                                        self.stepsize, 1, .02, .67)
            self.stepsize = stepsize
            self.ar = ar.mean().item()

        xy_fake = torch.cat([x, y_fake], dim=1)
        logp_fake = self(xy_fake)

        obj = logp_real.mean() - logp_fake.mean()
        loss = -obj + (logp_real ** 2).mean() * self.p_control + (logp_fake ** 2).mean() * self.n_control
        return {'loss': loss, 'real': logp_real.mean().item(), 'fake': logp_fake.mean().item()}

    def predict_fn(self, x, ymin, ymax, stepsize=.1, iters=100, n_samples=10, return_all=False):
        y_init = torch.arange(n_samples).float() / (n_samples - 1) * (ymax - ymin) + ymin
        y_init = y_init.repeat(x.size(0))[:, None].to(x.device)
        y = nn.Parameter(y_init)
        optim = torch.optim.Adam([y], stepsize)
        x_r = x.repeat_interleave(n_samples, dim=0)
        for i in range(iters):
            optim.zero_grad()
            xy = torch.cat([x_r, y], dim=1)
            logp_y_given_x = self(xy)
            loss = -logp_y_given_x.mean()
            loss.backward()
            optim.step()
        xy = torch.cat([x_r, y], dim=1)
        logp_y_given_x = self(xy)
        logp = logp_y_given_x.view(x.size(0), -1)
        y = y.view(x.size(0), -1)
        logp_max, max_ind = logp.max(dim=1)
        y_max = y[torch.arange(x.size(0)), max_ind]
        if return_all:
            return y
        else:
            return y_max.data

    def plot_energy_fn(self, x, y, ymin, ymax, n_samples=100):
        plt.clf()
        y_e = torch.arange(n_samples).float() / (n_samples - 1) * (ymax - ymin) + ymin
        y_e = y_e[:, None].to(x.device)
        x_r = x[None].repeat_interleave(n_samples, dim=0)
        xy = torch.cat([x_r, y_e], dim=1)
        logp_y_given_x = self(xy)
        logp_y_given_x = logp_y_given_x - logp_y_given_x.mean()
        p_y_given_x = logp_y_given_x.exp()
        p_y_given_x = p_y_given_x / p_y_given_x.sum()
        plt.plot(y_e.detach().cpu().numpy(), p_y_given_x.detach().cpu().numpy(), c='b')
        plt.plot([y.item(), y.item()], [0., max(p_y_given_x)], c='orange')

        y_preds = self.predict_fn(x[None], ymin, ymax, return_all=True)[0]
        for yp in y_preds:
            plt.plot([yp.item(), yp.item()], [0., max(p_y_given_x)], c='y')
        ypb = self.predict_fn(x[None], ymin, ymax)[0]
        plt.plot([ypb.item(), ypb.item()], [0., max(p_y_given_x)], c='g')

def get_data(args):
    if args.dataset == "concrete":
        training_data_x = pd.read_excel("data/concrete.xls")
        args.data_dim = 8
        x = training_data_x.to_numpy()
    elif args.dataset == "protein":
        training_data_x = pd.read_csv("data/protein.csv")
        args.data_dim = 9
        x = training_data_x.to_numpy()
    elif args.dataset == "power_plant":
        training_data_x = pd.read_excel("data/power_plant/Folds5x2_pp.xlsx")
        args.data_dim = 4
        x = training_data_x.to_numpy()
    elif args.dataset == "navy":
        with open("data/navy.txt", 'r') as f:
            lines = f.readlines()
            vals = []
            for line in lines:
                ls = line.split()
                ls = [float(v) for v in ls]
                vals.append(ls)
        x = np.array(vals).astype(np.float)
        x = x[:, :-1]  # take out last thing
        args.data_dim = 16
    elif args.dataset == "year":
        with open("data/year.txt", 'r') as f:
            lines = f.readlines()
            vals = []
            for line in lines:
                ls = line.strip().split(',')
                ls = [float(v) for v in ls]
                vals.append(ls)
        x = np.array(vals).astype(np.float)
        args.data_dim = 90
    else:
        raise ValueError


    print(x.shape)
    for i in range(x.shape[1]):
        plt.clf()
        plt.hist(x[:, i])
        plt.savefig("{}/features/feature_{}.png".format(args.save_dir, i))

    mu = x.mean(0)
    std = x.std(0)

    mu_t, std_t = torch.from_numpy(mu[None]).float()[0], torch.from_numpy(std[None]).float()[0]

    x = (x - mu[None]) / (std[None] + 1e-6)

    for i in range(x.shape[1]):
        plt.clf()
        plt.hist(x[:, i])
        plt.savefig("{}/features/normed_feature_{}.png".format(args.save_dir, i))
    init_dist = distributions.Normal(torch.zeros_like(mu_t), torch.ones_like(std_t))
    n_test = int(x.shape[0] * args.test_frac)
    if args.dataset == "year":
        n_test = 51630
    inds = list(range(x.shape[0]))
    np.random.seed(args.seed)
    np.random.shuffle(inds)

    train_inds = np.array(inds[:-n_test])
    unlabeled_inds = train_inds.copy()
    if args.n_labels != -1:
        train_inds = train_inds[:args.n_labels]
    test_inds = np.array(inds[-n_test:])
    train, test = x[train_inds], x[test_inds]
    unlab = x[unlabeled_inds]

    # concrete and power-plant is [x, y]
    if args.dataset == "concrete" or args.dataset == "power_plant" or args.dataset == "navy" or args.dataset == "year":
        xtr, ytr = train[:, :-1], train[:, -1]
        xte, yte = test[:, :-1], test[:, -1]
        xul = unlab[:, :-1]
        yind = -1
    # protein is [y, x]
    elif args.dataset == "protein":
        xtr, ytr = train[:, 1:], train[:, 0]
        xte, yte = test[:, 1:], test[:, 0]
        xul = unlab[:, 1:]
        yind = 0
    else:
        raise ValueError

    unnormalize = lambda y: y * std[yind] + mu[yind]

    ymin, ymax = min(ytr), max(ytr)

    xtr, ytr, xte, yte, xul = [torch.from_numpy(v).float() for v in [xtr, ytr, xte, yte, xul]]

    dset_train = torch.utils.data.TensorDataset(xtr, ytr)
    dset_test = torch.utils.data.TensorDataset(xte, yte)

    print(len(dset_train), len(dset_test), unlab.shape)
    if args.buffer_size == -1:
        args.buffer_size = len(dset_train)
    print(ymin, ymax)

    dload_train = torch.utils.data.DataLoader(dset_train, args.batch_size, True, drop_last=True)
    dload_test = torch.utils.data.DataLoader(dset_test, args.batch_size, True, drop_last=False)
    return dload_train, dload_test, init_dist, unnormalize, yind, ymin, ymax, xul


def main(args):
    utils.makedirs(args.save_dir)
    utils.makedirs("{}/figs_test".format(args.save_dir))
    utils.makedirs("{}/figs_train".format(args.save_dir))
    utils.makedirs("{}/features".format(args.save_dir))
    logf = open("{}/log.txt".format(args.save_dir), 'w')
    dload_train, dload_test, init_dist, unnormalize, yind, ymin, ymax, Xul = get_data(args)
    hd = args.h_dim
    in_dim = args.data_dim + 1 if args.model in ("ebm", "ebr") else args.data_dim
    layers = [
        nn.Linear(in_dim, hd),
        nn.LeakyReLU(.2),
    ]
    for l in range(args.h_layers):
        layers.append(nn.Linear(hd, hd))
        layers.append(nn.LeakyReLU(.2))
    layers.append(nn.Linear(hd, 1))

    net = nn.Sequential(*layers)

    if args.model == "regression":
        predict_fn = lambda x, **kwargs: net(x).squeeze()
        loss_fn = lambda x, y, **kwargs: {'loss': (predict_fn(x) - y) ** 2}
    else:
        net = EBM(net, yind, init_dist, args.base_dist,
                  args.sgld_lr, args.sgld_std, args.mcmc_steps, args.sample_init,
                  args.p_control, args.n_control,
                  args.buffer_size, args.reinit_freq, args.sampler)
        predict_fn = lambda x: net.predict_fn(x, ymin, ymax)
        if args.model == "ebm":
            loss_fn = net.logpxy_objective
        elif args.model == "ebr":
            loss_fn = net.logpy_given_x_objective


    rmse_fn = lambda l: torch.cat(l).mean() ** .5
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    net.to(device)

    train_stats = []
    test_stats = []
    train_loss = []
    reals, fakes = [], []
    for epoch in range(args.n_epochs):
        rvals = None
        sq_err_train = []
        tl = []
        r, f = [], []
        for i, (x, y) in enumerate(dload_train):
            x = x.to(device)
            x = x + torch.randn_like(x) * args.data_noise
            y = y.to(device)
            y = y# + torch.randn_like(y) * args.data_noise

            rvals = loss_fn(x, y)
            all_loss = rvals['loss']
            tl.append(all_loss.mean().item())
            if 'real' in rvals:
                r.append(rvals['real'])
                f.append(rvals['fake'])
            loss = all_loss.mean()
            if args.logpx_weight > 0:
                inds = list(range(Xul.size(0)))
                inds = np.random.choice(inds, x.size(0), replace=False)
                inds = torch.from_numpy(inds)
                xul = Xul[inds].to(device)
                ulrvals = net.logpx_objective(xul)
                loss += ulrvals['loss'].mean() * args.logpx_weight
            else:
                ulrvals = None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.viz_every == 0:
                this_sq = (unnormalize(y) - unnormalize(predict_fn(x))) ** 2
                sq_err_train.append(this_sq)
                rmse_train = rmse_fn([this_sq])
                print("    {} | {} | rmse train {}".format(epoch, i, rmse_train.item()))
                if args.model in ("ebr", "ebm"):
                    print("    {} | {} | real = {:.4f}, fake = {:.4f}, diff = {:.4f}".format(epoch, i, rvals['real'],
                                                                                         rvals['fake'],
                                                                                         rvals['real'] - rvals['fake']))
                    if ulrvals is not None:
                        print(
                            "    UNLABELED {} | {} | real = {:.4f}, fake = {:.4f}, diff = {:.4f}".format(epoch, i, ulrvals['real'],
                                                                                               ulrvals['fake'],
                                                                                               ulrvals['real'] - ulrvals[
                                                                                                   'fake']))
                    if args.sampler == "hmc":
                        print("    stepsize = {}, ar = {}".format(net.stepsize, net.ar))
                    net.plot_energy_fn(x[0], y[0], ymin, ymax)
                    plt.savefig("{}/figs_train/epoch_{}_itr_{}.png".format(args.save_dir, epoch, i))
        train_loss.append(np.mean(tl))
        reals.append(np.mean(r))
        fakes.append(np.mean(f))

        sq_err = []
        for i, (x, y) in enumerate(tqdm(dload_test)):
            x = x.to(device)
            x = x + torch.randn_like(x) * args.data_noise
            y = y.to(device)
            #y = y + torch.randn_like(y) * args.data_noise

            loss = ((unnormalize(y) - unnormalize(predict_fn(x))) ** 2)
            sq_err.append(loss)
            if args.test_iters != -1 and i > args.test_iters:
                break

        rmse_test = rmse_fn(sq_err)
        rmse_train = rmse_fn(sq_err_train)
        train_stats.append(rmse_train.item())
        test_stats.append(rmse_test.item())

        plt.clf()
        plt.plot(train_stats, c='b', label='train')
        plt.plot(test_stats, c='r', label='test')
        plt.legend()
        plt.savefig("{}/rmse.png".format(args.save_dir))

        plt.clf()
        plt.plot(train_loss, c='b', label='train')
        plt.legend()
        plt.savefig("{}/loss.png".format(args.save_dir))

        plt.clf()
        plt.plot(reals, c='b', label='real')
        plt.plot(fakes, c='r', label='fake')
        plt.legend()
        plt.savefig("{}/logpx.png".format(args.save_dir))

        logs = "Epoch {} | rmse train {}, rmse test {}".format(epoch, rmse_train.item(), rmse_test.item())
        print(logs)
        logf.write(logs + '\n')
        if args.model in ("ebr", "ebm"):
            logs = "{} | {} | real = {:.4f}, fake = {:.4f}, diff = {:.4f}".format(epoch, i,
                                                                                  rvals['real'], rvals['fake'],
                                                                                  rvals['real'] - rvals['fake'])
            print(logs)
            logf.write(logs + '\n')
            net.plot_energy_fn(x[0], y[0], ymin, ymax)
            plt.savefig("{}/figs_test/epoch_{}.png".format(args.save_dir, epoch))







if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="concrete")
    parser.add_argument("--loss", type=str, default="ml")
    parser.add_argument("--model", type=str, default="regression", choices=["ebm", "ebr", "regression"])
    parser.add_argument("--sample_init", type=str, default="cd", choices=["cd", "pcd", "short-run"])
    parser.add_argument("--sampler", type=str, default="sgld", choices=["sgld", "hmc"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--logpx_weight", type=float, default=0.)
    parser.add_argument("--test_frac", type=float, default=.1)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_labels", type=int, default=-1)
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--h_layers", type=int, default=1)
    parser.add_argument("--mcmc_steps", type=int, default=25)
    # regularization
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--p_control", type=float, default=0.0)
    parser.add_argument("--n_control", type=float, default=0.0)
    # EBM specific
    parser.add_argument("--data_noise", type=float, default=.0)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='/tmp/regression')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=20, help="Iterations between print")
    parser.add_argument("--viz_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--semi-supervised", type=bool, default=False)
    parser.add_argument("--sgld_std", type=float, default=.1)
    parser.add_argument("--sgld_lr", type=float, default=-1)
    parser.add_argument("--base_dist", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--seed", type=int, default=1234, help="Iterations between print")
    parser.add_argument("--test_iters", type=int, default=-1, help="Iterations between print")
    # PCD
    parser.add_argument("--buffer_size", type=int, default=-1, help="Iterations between print")
    parser.add_argument("--reinit_freq", type=float, default=0.0)

    args = parser.parse_args()
    main(args)

"""
 python ebm_regression.py --dataset concrete --save_dir /scratch/gobi1/gwohl/conc_cd --lr .0003 --data_noise .01 --weight_decay .0005 --batch_size 128 --n_epochs 250 --p_control 0. --n_control 0. --n_labels 128 --viz_every 20 --h_dim 2000 --mcmc_steps 25 --sample_init pcd --model ebm --sampler hmc --h_layers 2 --logpx_weight 1.0 --buffer^Cize 5000 --reinit_freq .05 --base_dist
"""