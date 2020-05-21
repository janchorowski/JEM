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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stepsize = 1.

def get_data(args):
    if args.dataset == "concrete":
        import pandas as pd
        training_data_x = pd.read_excel("data/concrete.xls")
        x = training_data_x.as_matrix()

        mu = x.mean(0)[None]
        std = x.std(0)[None]

        mu_t, std_t = torch.from_numpy(mu).float()[0], torch.from_numpy(std).float()[0]

        if args.normalize:
            x = (x - mu) / std
            init_dist = distributions.Normal(
                torch.zeros_like(mu_t), torch.ones_like(std_t))
        else:
            init_dist = distributions.Normal(mu_t, std_t)
            mu = np.zeros_like(mu)
            std = np.ones_like(std)

        n_test = int(x.shape[0] * args.test_frac)
        inds = list(range(x.shape[0]))
        np.random.shuffle(inds)

        train_inds = np.array(inds[:-n_test])
        test_inds = np.array(inds[-n_test:])
        train, test = x[train_inds], x[test_inds]

        xtr, ytr = train[:, :-1], train[:, -1]
        xte, yte = test[:, :-1], test[:, -1]

        xtr, ytr, xte, yte = [torch.from_numpy(v).float() for v in [xtr, ytr, xte, yte]]

        dset_train = torch.utils.data.TensorDataset(xtr, ytr)
        dset_test = torch.utils.data.TensorDataset(xte, yte)

        print(len(dset_train), len(dset_test))

        dload_train = torch.utils.data.DataLoader(dset_train, args.batch_size, True, drop_last=True)
        dload_test = torch.utils.data.DataLoader(dset_test, args.batch_size, False, drop_last=False)

        args.data_dim = 8
        return dload_train, dload_test, torch.from_numpy(mu).float(), torch.from_numpy(std).float(), init_dist

    elif args.dataset == "protein":
        pass
    elif args.dataset == "power_plant":
        pass
    else:
        raise NotImplementedError

def main(args):
    dload_train, dload_test, mu, std, init_dist = get_data(args)
    hd = 50
    in_dim = args.data_dim + 1 if args.ebm or args.ebr else args.data_dim
    net = nn.Sequential(
        nn.Linear(in_dim, hd),
        #nn.ELU(),
        nn.LeakyReLU(.2),
        nn.Linear(hd, hd),
        #nn.ELU(),
        nn.LeakyReLU(.2),
        nn.Linear(hd ,hd),
        #nn.ELU(),
        nn.LeakyReLU(.2),
        nn.Linear(hd, 1)
    )

    unnormalize = lambda y: y * std[:, -1] + mu[:, -1]

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stepsize = 1. / 10


    if args.ebm:
        def loss_fn(x, y, stepsize=1.):
            xy = torch.cat([x, y[:, None]], dim=1)
            logp_y_given_x_real = net(xy)

            xy_init = init_dist.sample_n(x.size(0))
            log_fn = lambda xy: net(xy).squeeze()
            xy_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, xy_init, 5, 1, 5, stepsize, 1, .02, .67)
            logp_y_given_x_fake = net(xy_fake)
            obj = logp_y_given_x_real.mean() - logp_y_given_x_fake.mean()
            loss = -obj
            # print(logp_y_given_x_real.mean(), logp_y_given_x_fake.mean())
            return {'loss': loss, 'stepsize': stepsize}

        def predict_fn(x, stepsize=1., iters=10):
            y_init = init_dist.sample_n(x.size(0))[:, -1:]
            y = nn.Parameter(y_init)
            optim = torch.optim.Adam([y], stepsize)
            for i in range(iters):
                optim.zero_grad()
                xy = torch.cat([x, y], dim=1)
                logp_y_given_x = net(xy)
                loss = -logp_y_given_x.mean()
                loss.backward()
                optim.step()
            return y.data[:, 0]
    elif args.ebr:
        if args.loss == "ml":
            def loss_fn(x, y, stepsize=1.):
                xy = torch.cat([x, y[:, None]], dim=1)
                logp_y_given_x_real = net(xy)

                y_init = init_dist.sample_n(x.size(0))[:, -1:]
                log_fn = lambda y: net(torch.cat([x, y], dim=1)).squeeze()
                y_fake, ar, stepsize = hmc.get_ebm_samples(log_fn, y_init, 15, 1, 5, stepsize, 1, .02, .67)
                xy_fake = torch.cat([x, y_fake], dim=1)
                logp_y_given_x_fake = net(xy_fake)
                obj = logp_y_given_x_real.mean() - logp_y_given_x_fake.mean()
                loss = -obj + (logp_y_given_x_real ** 2).mean() * .5 + (logp_y_given_x_fake ** 2).mean() * .5
                #print()
                return {'loss': loss, 'stepsize': stepsize,
                        'real': logp_y_given_x_real.mean(), 'fake': logp_y_given_x_fake.mean()}
        else:
            def loss_fn(x, y, **kwargs):
                xy = torch.cat([x, y[:, None]], dim=1)
                logp_y_given_x_real = net(xy)

                y_fake = init_dist.sample_n(x.size(0))[:, -1:]
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


        def predict_fn(x, stepsize=1., iters=10):
            y_init = init_dist.sample_n(x.size(0))[:, -1:]
            y = nn.Parameter(y_init)
            optim = torch.optim.Adam([y], stepsize)
            for i in range(iters):
                optim.zero_grad()
                xy = torch.cat([x, y], dim=1)
                logp_y_given_x = net(xy)
                loss = -logp_y_given_x.mean()
                loss.backward()
                optim.step()
            return y.data[:, 0]


    else:
        predict_fn = lambda x, **kwargs: net(x).squeeze()
        loss_fn = lambda x, y, **kwargs: {'loss': (predict_fn(x) - y) ** 2, "stepsize": 1.}

    for epoch in range(args.n_epochs):
        rvals = None
        sq_err_train = []
        for i, (x, y) in enumerate(dload_train):
            x = x.to(device)
            x = x + torch.randn_like(x) * .01
            y = y.to(device)

            rvals = loss_fn(x, y, stepsize=stepsize)
            all_loss = rvals['loss']
            stepsize = rvals['stepsize']
            loss = all_loss.mean()
            sq_err_train.append((unnormalize(y) - unnormalize(predict_fn(x))) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        sq_err = []
        for i, (x, y) in enumerate(dload_test):
            x = x.to(device)
            x = x + torch.randn_like(x) * .01
            y = y.to(device)

            loss = ((unnormalize(y) - unnormalize(predict_fn(x, stepsize=1.))) ** 2)
            sq_err.append(loss)
        sq_err = torch.cat(sq_err)
        mse = sq_err.mean()
        rmse = mse ** .5

        sq_err_train = torch.cat(sq_err_train)
        mse_train = sq_err_train.mean()
        rmse_train = mse_train ** .5

        print("{} | {} | rmse train {}, rmse test {}, stepsize = {}".format(epoch, i, rmse_train.item(), rmse.item(), stepsize))
        print("{} | {} | real = {:.4f}, fake = {:.4f}, diff = {:.4f}".format(epoch, i, rvals['real'], rvals['fake'],
                                                                             rvals['real'] - rvals['fake']))







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
    parser.add_argument("--ebm", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--ebr", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--normalize", action="store_true", help="Run VAT instead of JEM")

    args = parser.parse_args()
    main(args)