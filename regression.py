import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as distributions
import torchvision
import torchvision.transforms as transforms
import torch.utils
import torch.utils.data
import hmc
import pandas as pd
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stepsize = 1.

def get_data(args):
    if args.dataset == "concrete":
        training_data_x = pd.read_excel("data/concrete.xls")
        args.data_dim = 8
        x = training_data_x.to_numpy()
    elif args.dataset == "protein":
        training_data_x = pd.read_csv("data/protein.csv")
        args.data_dim = 9
    elif args.dataset == "power_plant":
        training_data_x = pd.read_excel("data/power_plant/Folds5x2_pp.xlsx")
        args.data_dim = 4
    else:
        raise ValueError

    x = training_data_x.to_numpy()
    # print(x.shape)
    # for i in range(x.shape[1]):
    #     plt.hist(x[:, i])
    #     plt.show()

    mu = x.mean(0)
    std = x.std(0)

    mu_t, std_t = torch.from_numpy(mu[None]).float()[0], torch.from_numpy(std[None]).float()[0]

    x = (x - mu[None]) / (std[None] + 1e-6)
    init_dist = distributions.Normal(torch.zeros_like(mu_t), torch.ones_like(std_t))

    n_test = int(x.shape[0] * args.test_frac)
    inds = list(range(x.shape[0]))
    np.random.shuffle(inds)

    train_inds = np.array(inds[:-n_test])
    test_inds = np.array(inds[-n_test:])
    train, test = x[train_inds], x[test_inds]

    # concrete and power-plant is [x, y]
    if args.dataset == "concrete" or args.dataset == "power_plant":
        xtr, ytr = train[:, :-1], train[:, -1]
        xte, yte = test[:, :-1], test[:, -1]
        unnormalize = lambda y: y * std[-1] + mu[-1]
        yind = -1
    # protein is [y, x]
    elif args.dataset == "protein":
        xtr, ytr = train[:, 1:], train[:, 0]
        xte, yte = test[:, 1:], test[:, 0]
        unnormalize = lambda y: y * std[0] + mu[0]
        yind = 0
    else:
        raise ValueError

    ymin, ymax = min(ytr), max(ytr)

    xtr, ytr, xte, yte = [torch.from_numpy(v).float() for v in [xtr, ytr, xte, yte]]

    dset_train = torch.utils.data.TensorDataset(xtr, ytr)
    dset_test = torch.utils.data.TensorDataset(xte, yte)

    print(len(dset_train), len(dset_test))
    print(ymin, ymax)

    dload_train = torch.utils.data.DataLoader(dset_train, args.batch_size, True, drop_last=True)
    dload_test = torch.utils.data.DataLoader(dset_test, args.batch_size, False, drop_last=False)
    return dload_train, dload_test, init_dist, unnormalize, yind, ymin, ymax


def main(args):
    utils.makedirs(args.save_dir)
    utils.makedirs("{}/figs_test".format(args.save_dir))
    utils.makedirs("{}/figs_train".format(args.save_dir))
    dload_train, dload_test, init_dist, unnormalize, yind, ymin, ymax = get_data(args)
    hd = args.h_dim
    in_dim = args.data_dim + 1 if args.ebm or args.ebr else args.data_dim
    net = nn.Sequential(
        nn.Linear(in_dim, hd),
        #nn.ELU(),
        nn.LeakyReLU(.2),
        #nn.ReLU(),
        nn.Linear(hd, hd),
        #nn.ELU(),
        nn.LeakyReLU(.2),
        #nn.ReLU(),
        nn.Linear(hd ,hd),
        #nn.ELU(),
        nn.LeakyReLU(.2),
        #nn.ReLU(),
        nn.Linear(hd, 1)
    )



    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stepsize = 1. / 10

    rmse_fn = lambda l: torch.cat(l).mean() ** .5
    if args.ebm:
        def loss_fn(x, y, stepsize=1.):
            xy = torch.cat([x, y[:, None]], dim=1)
            logp_y_given_x_real = net(xy)

            xy_init = init_dist.sample((x.size(0),))#sample_n(x.size(0))
            log_fn = lambda xy: net(xy).squeeze()
            xy_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, xy_init, args.mcmc_steps, 1, 5, stepsize, 1, .02, .67)
            logp_y_given_x_fake = net(xy_fake)
            obj = logp_y_given_x_real.mean() - logp_y_given_x_fake.mean()
            loss = -obj
            # print(logp_y_given_x_real.mean(), logp_y_given_x_fake.mean())
            return {'loss': loss, 'stepsize': stepsize,
                    'real': logp_y_given_x_real.mean(), 'fake': logp_y_given_x_fake.mean()}

        def predict_fn(x, stepsize=.1, iters=100, n_samples=10, return_all=False):
            y_init = torch.arange(n_samples).float() / (n_samples - 1) * (ymax - ymin) + ymin
            y_init = y_init.repeat(x.size(0))[:, None]
            y = nn.Parameter(y_init)
            optim = torch.optim.Adam([y], stepsize)
            x_r = x.repeat_interleave(n_samples, dim=0)
            for i in range(iters):
                optim.zero_grad()
                xy = torch.cat([x_r, y], dim=1)
                logp_y_given_x = net(xy)
                loss = -logp_y_given_x.mean()
                loss.backward()
                optim.step()
            xy = torch.cat([x_r, y], dim=1)
            logp_y_given_x = net(xy)
            logp = logp_y_given_x.view(x.size(0), -1)
            y = y.view(x.size(0), -1)
            logp_max, max_ind = logp.max(dim=1)
            y_max = y[torch.arange(x.size(0)), max_ind]
            if return_all:
                return y
            else:
                return y_max.data

        def plot_energy_fn(x, y, n_samples=100):
            plt.clf()
            y_e = torch.arange(n_samples).float() / (n_samples - 1) * (ymax - ymin) + ymin
            y_e = y_e[:, None]
            x_r = x[None].repeat_interleave(n_samples, dim=0)
            xy = torch.cat([x_r, y_e], dim=1)
            logp_y_given_x = net(xy)
            p_y_given_x = logp_y_given_x.exp()
            p_y_given_x = p_y_given_x / p_y_given_x.sum()
            plt.plot(y_e.detach().cpu().numpy(), p_y_given_x.detach().cpu().numpy(), c='b')
            plt.plot([y.item(), y.item()], [0., max(p_y_given_x)], c='orange')
            y_preds = predict_fn(x[None], return_all=True)[0]
            for yp in y_preds:
                plt.plot([yp.item(), yp.item()], [0., max(p_y_given_x)], c='r')
            y_preds = predict_fn(x[None], return_all=True)[0]
            for yp in y_preds:
                plt.plot([yp.item(), yp.item()], [0., max(p_y_given_x)], c='y')
            ypb = predict_fn(x[None])[0]
            plt.plot([ypb.item(), ypb.item()], [0., max(p_y_given_x)], c='g')

    elif args.ebr:
        if args.loss == "ml":
            def loss_fn(x, y, stepsize=1.):
                xy = torch.cat([x, y[:, None]], dim=1)
                logp_y_given_x_real = net(xy)

                y_init = init_dist.sample((x.size(0),))[:, yind][:, None]#.sample_n(x.size(0))[:, yind][:, None]
                log_fn = lambda y: net(torch.cat([x, y], dim=1)).squeeze()
                y_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, y_init, args.mcmc_steps, 1, 5, stepsize, 1, .02, .67)
                xy_fake = torch.cat([x, y_fake], dim=1)
                logp_y_given_x_fake = net(xy_fake)
                obj = logp_y_given_x_real.mean() - logp_y_given_x_fake.mean()
                loss = -obj + (logp_y_given_x_real ** 2).mean() * args.p_control #* .5 + (logp_y_given_x_fake ** 2).mean() * .5
                return {'loss': loss, 'stepsize': stepsize,
                        'real': logp_y_given_x_real.mean(), 'fake': logp_y_given_x_fake.mean()}
        else:
            def loss_fn(x, y, **kwargs):
                xy = torch.cat([x, y[:, None]], dim=1)
                logp_y_given_x_real = net(xy)

                y_fake = init_dist.sample((x.size(0),))[:, -1:]#sample_n(x.size(0))[:, -1:]
                xy_fake = torch.cat([x, y_fake], dim=1)
                logp_y_given_x_fake = net(xy_fake)

                logpny = init_dist.log_prob(xy)[:, -1]
                logpny_fake = init_dist.log_prob(xy_fake)[:, -1]

                logits_real = logp_y_given_x_real - logpny
                logits_fake = logp_y_given_x_fake - logpny_fake

                loss_real = nn.BCEWithLogitsLoss()(logits_real, torch.ones_like(logits_real))
                loss_fake = nn.BCEWithLogitsLoss()(logits_fake, torch.zeros_like(logits_fake))
                loss = loss_real + loss_fake
                return {'loss': loss, 'stepsize': stepsize}


        def predict_fn(x, stepsize=.1, iters=100, n_samples=10, return_all=False):
            y_init = torch.arange(n_samples).float() / (n_samples - 1) * (ymax - ymin) + ymin
            y_init = y_init.repeat(x.size(0))[:, None]
            y = nn.Parameter(y_init)
            optim = torch.optim.Adam([y], stepsize)
            x_r = x.repeat_interleave(n_samples, dim=0)
            for i in range(iters):
                optim.zero_grad()
                xy = torch.cat([x_r, y], dim=1)
                logp_y_given_x = net(xy)
                loss = -logp_y_given_x.mean()
                loss.backward()
                optim.step()
            xy = torch.cat([x_r, y], dim=1)
            logp_y_given_x = net(xy)
            logp = logp_y_given_x.view(x.size(0), -1)
            y = y.view(x.size(0), -1)
            logp_max, max_ind = logp.max(dim=1)
            y_max = y[torch.arange(x.size(0)), max_ind]
            if return_all:
                return y
            else:
                return y_max.data

        def plot_energy_fn(x, y, n_samples=100):
            plt.clf()
            y_e = torch.arange(n_samples).float() / (n_samples - 1) * (ymax - ymin) + ymin
            y_e = y_e[:, None]
            x_r = x[None].repeat_interleave(n_samples, dim=0)
            xy = torch.cat([x_r, y_e], dim=1)
            logp_y_given_x = net(xy)
            p_y_given_x = logp_y_given_x.exp()
            p_y_given_x = p_y_given_x / p_y_given_x.sum()
            plt.plot(y_e.detach().cpu().numpy(), p_y_given_x.detach().cpu().numpy(), c='b')
            plt.plot([y.item(), y.item()], [0., max(p_y_given_x)], c='orange')
            y_preds = predict_fn(x[None], return_all=True)[0]
            for yp in y_preds:
                plt.plot([yp.item(), yp.item()], [0., max(p_y_given_x)], c='r')
            y_preds = predict_fn(x[None], return_all=True)[0]
            for yp in y_preds:
                plt.plot([yp.item(), yp.item()], [0., max(p_y_given_x)], c='y')
            ypb = predict_fn(x[None])[0]
            plt.plot([ypb.item(), ypb.item()], [0., max(p_y_given_x)], c='g')


    else:
        predict_fn = lambda x, **kwargs: net(x).squeeze()
        loss_fn = lambda x, y, **kwargs: {'loss': (predict_fn(x) - y) ** 2, "stepsize": 1.}

    train_stats = []
    test_stats = []
    for epoch in range(args.n_epochs):
        rvals = None
        sq_err_train = []
        for i, (x, y) in enumerate(dload_train):
            x = x.to(device)
            x = x + torch.randn_like(x) * args.data_noise
            y = y.to(device)

            rvals = loss_fn(x, y, stepsize=stepsize)
            all_loss = rvals['loss']
            stepsize = rvals['stepsize']
            loss = all_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.viz_every == 0:
                this_sq = (unnormalize(y) - unnormalize(predict_fn(x))) ** 2
                sq_err_train.append(this_sq)
                rmse_train = rmse_fn([this_sq])
                print("    {} | {} | rmse train {}, stepsize = {}".format(epoch, i, rmse_train.item(), stepsize))
                if args.ebm or args.ebr:
                    print("    {} | {} | real = {:.4f}, fake = {:.4f}, diff = {:.4f}".format(epoch, i, rvals['real'],
                                                                                         rvals['fake'],
                                                                                         rvals['real'] - rvals['fake']))
                    plot_energy_fn(x[0], y[0])
                    plt.savefig("{}/figs_train/epoch_{}_itr_{}.png".format(args.save_dir, epoch, i))

        sq_err = []
        for i, (x, y) in enumerate(dload_test):
            x = x.to(device)
            x = x + torch.randn_like(x) * args.data_noise
            y = y.to(device)

            loss = ((unnormalize(y) - unnormalize(predict_fn(x, stepsize=1.))) ** 2)
            sq_err.append(loss)

        rmse_test = rmse_fn(sq_err)
        rmse_train = rmse_fn(sq_err_train)
        train_stats.append(rmse_train.item())
        test_stats.append(rmse_test.item())
        plt.clf()
        plt.plot(train_stats, c='b', label='train')
        plt.plot(test_stats, c='r', label='test')
        plt.legend()
        plt.savefig("{}/rmse.png".format(args.save_dir))

        print("{} | {} | rmse train {}, rmse test {}, stepsize = {}".format(epoch, i,
                                                                            rmse_train.item(), rmse_test.item(),
                                                                            stepsize))
        if args.ebm or args.ebr:
            print("{} | {} | real = {:.4f}, fake = {:.4f}, diff = {:.4f}".format(epoch, i, rvals['real'], rvals['fake'],
                                                                                 rvals['real'] - rvals['fake']))
            plot_energy_fn(x[0], y[0])
            plt.savefig("{}/figs_test/epoch_{}.png".format(args.save_dir, epoch))







if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="concrete")#, choices=["mnist", "moons", "circles"])
    parser.add_argument("--loss", type=str, default="concrete")#, choices=["mnist", "moons", "circles"])
    parser.add_argument("--mode", type=str, default="reverse_kl")
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_frac", type=float, default=.1)
    parser.add_argument("--v_norm", type=float, default=.01)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--labels_per_class", type=int, default=10,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--mcmc_steps", type=int, default=25)
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
    parser.add_argument("--ebm", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--ebr", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--normalize", action="store_true", help="Run VAT instead of JEM")

    args = parser.parse_args()
    main(args)