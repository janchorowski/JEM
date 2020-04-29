import argparse
import torch
import torch.nn as nn
import torch.utils
import torch.distributions as distributions
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sklearn.datasets as datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
import utils


def main(args):
    logp_net = nn.Sequential(
        nn.Linear(args.data_dim, args.h_dim),
        nn.LeakyReLU(.2),
        nn.Linear(args.h_dim, args.h_dim),
        nn.LeakyReLU(.2),
        nn.Linear(args.h_dim, 1)
    )
    logp_fn = lambda x: logp_net(x)# - (x * x).flatten(start_dim=1).sum(1)

    generator = nn.Sequential(
        nn.Linear(args.noise_dim, args.h_dim),
        nn.ReLU(),
        nn.BatchNorm1d(args.h_dim),
        nn.Linear(args.h_dim, args.h_dim),
        nn.ReLU(),
        nn.BatchNorm1d(args.h_dim),
        nn.Linear(args.h_dim, args.data_dim)
    )
    logsigma = nn.Parameter(torch.zeros(1,))

    params = list(logp_net.parameters()) + list(generator.parameters()) + [logsigma]

    optimizer = torch.optim.Adam(params, lr=args.lr, betas=[.5, .9], weight_decay=args.weight_decay)

    train_loader, test_loader = get_data(args)

    def sample_q(n):
        h = torch.randn((n, args.noise_dim)).to(device)
        x_mu = generator(h)
        x = x_mu + torch.randn_like(x_mu) * logsigma.exp()
        return x, h

    def logq_unnorm(x, h):
        logph = distributions.Normal(0, 1).log_prob(h).sum(1)
        px_given_h = distributions.Normal(generator(h), logsigma.exp())
        logpx_given_h = px_given_h.log_prob(x).sum(1)
        return logpx_given_h + logph


    def refine_q(x_init, h_init, n_steps, sgld_step):
        x_k = torch.clone(x_init).requires_grad_()
        h_k = torch.clone(h_init).requires_grad_()
        sgld_sigma = (2 * sgld_step) ** .5
        for k in range(n_steps):
            logp = logp_fn(x_k)[:, 0]
            logq = logq_unnorm(x_k, h_k)
            g_h = torch.autograd.grad(logq.sum(), [h_k], retain_graph=True)[0]

            # sample h tilde ~ q(h|x_k)
            h_tilde = h_k + g_h * sgld_step + torch.randn_like(h_k) * sgld_sigma
            h_tilde = h_tilde.detach()

            logq_tilde = logq_unnorm(x_k, h_tilde)
            g_x = torch.autograd.grad(logp.sum() + logq.sum() - logq_tilde.sum(), [x_k], retain_graph=True)[0]
            # x update
            x_k = x_k + g_x * sgld_step + torch.randn_like(x_k) * sgld_sigma
            x_k = x_k.detach().requires_grad_()

            # h update
            h_k = h_k + g_h * sgld_step + torch.randn_like(h_k) * sgld_sigma
            h_k = h_k.detach().requires_grad_()

        return x_k.detach(), h_k.detach()

    itr = 0
    for epoch in range(args.n_epochs):
        for x_d, _ in train_loader:
            optimizer.zero_grad()
            x_d = x_d.to(device)

            x_init, h_init = sample_q(args.batch_size)
            x, h = refine_q(x_init, h_init, args.n_steps, args.sgld_step)

            logp_obj = (logp_fn(x_d) - logp_fn(x.detach()))[:, 0].mean()
            logq_obj = logq_unnorm(x.detach(), h.detach()).mean()

            loss = -(logp_obj + logq_obj)
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("({}) | log p obj = {:.4f}, log q obj = {:.4f}, sigma = {:.4f}".format(
                    itr, logp_obj.item(), logq_obj.item(), logsigma.exp().item()))

            if itr % args.viz_every == 0:
                plt.clf()
                xm = x.cpu().numpy()
                xn = x_d.cpu().numpy()
                xi = x_init.detach().cpu().numpy()
                ax = plt.subplot(1, 4, 1, aspect="equal", title='refined')
                ax.scatter(xm[:, 0], xm[:, 1])
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                ax = plt.subplot(1, 4, 2, aspect="equal", title='initial')
                ax.scatter(xi[:, 0], xi[:, 1])
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                ax = plt.subplot(1, 4, 3, aspect="equal", title='data')
                ax.scatter(xn[:, 0], xn[:, 1])
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)


                ax = plt.subplot(1, 4, 4, aspect="equal")
                utils.plt_flow_density(logp_fn, ax, low=-2, high=2)
                plt.savefig("/tmp/{}.png".format(itr))

            itr += 1



def get_data(args):
    if args.dataset == "moons":
        data, labels = datasets.make_moons(n_samples=10000, noise=.1)
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        dset = TensorDataset(data, labels)
        dload = DataLoader(dset, args.batch_size, True, drop_last=True)
        return dload, dload
    if args.dataset == "circles":
        data, labels = datasets.make_circles(n_samples=10000, noise=.1)
        data = torch.from_numpy(data).float()
        labels = torch.from_numpy(labels).long()
        dset = TensorDataset(data, labels)
        dload = DataLoader(dset, args.batch_size, True, drop_last=True)
        return dload, dload
    else:
        raise NotImplementedError




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="moons", choices=["mnist", "moons", "circles"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--labels_per_class", type=int, default=10,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--h_dim", type=int, default=100)
    parser.add_argument("--noise_dim", type=int, default=2)
    # loss weighting
    parser.add_argument("--p_x_weight", type=float, default=1.)
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    # regularization
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=10,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--sgld_step", type=float, default=.001)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=20, help="Iterations between print")
    parser.add_argument("--viz_every", type=int, default=200, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--semi-supervised", type=bool, default=False)
    parser.add_argument("--vat", action="store_true", help="Run VAT instead of JEM")

    args = parser.parse_args()
    if args.dataset == "moons" or args.dataset == "circles":
        args.data_dim = 2
    elif args.dataset == "mnist":
        args.data_di = 784

    main(args)
