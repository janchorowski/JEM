import argparse
import torch
import torch.nn as nn
import torch.utils
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
TOY_DSETS = ("moons", "circles", "8gaussians", "pinwheel", "2spirals", "checkerboard", "rings", "swissroll")


def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)

def main(args):
    utils.makedirs(args.save_dir)
    if args.dataset in TOY_DSETS:
        logp_net = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(args.data_dim, args.h_dim)),
            nn.LeakyReLU(.2),
            nn.utils.weight_norm(nn.Linear(args.h_dim, args.h_dim)),
            nn.LeakyReLU(.2),
            nn.Linear(args.h_dim, 1)
        )
        logp_fn = lambda x: logp_net(x)# - (x * x).flatten(start_dim=1).sum(1)/10
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
                self.logsigma = nn.Parameter(torch.zeros(1,))
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
        logp_fn = lambda x: logp_net(x)  # - (x * x).flatten(start_dim=1).sum(1)/10

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
                # self.logsigma = nn.Parameter((torch.ones(1,) * (args.sgld_step * 2)**.5).log(), requires_grad=False)
                #self.logsigma = nn.Parameter(torch.zeros(1, ) - 5)
                self.logsigma = nn.Parameter((torch.ones(1,) * .1).log(), requires_grad=False)


    g = G()

    params = list(logp_net.parameters()) + list(g.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, betas=[.0, .9], weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    train_loader, test_loader = get_data(args)

    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: torchvision.utils.save_image(torch.clamp(x, 0, 1), p, normalize=False, nrow=sqrt(x.size(0)))

    def sample_q(n):
        h = torch.randn((n, args.noise_dim)).to(device)
        x_mu = g.generator(h)
        x = x_mu + torch.randn_like(x_mu) * g.logsigma.exp()
        return x, h

    def logq_unnorm(x, h):
        logph = distributions.Normal(0, 1).log_prob(h).sum(1)
        px_given_h = distributions.Normal(g.generator(h), g.logsigma.exp())
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
            logq = logq_unnorm(x_k, h_k)
            g_x = torch.autograd.grad(logp.sum() + logq.sum() - logq_tilde.sum(), [x_k], retain_graph=True)[0]

            # x update
            x_k = x_k + g_x * sgld_step + torch.randn_like(x_k) * sgld_sigma
            x_k = x_k.detach().requires_grad_()#.clamp(0, 1)

            # h update
            logq = logq_unnorm(x_k, h_k)
            g_h = torch.autograd.grad(logq.sum(), [h_k], retain_graph=True)[0]
            h_k = h_k + g_h * sgld_step + torch.randn_like(h_k) * sgld_sigma
            h_k = h_k.detach().requires_grad_()

        # return x_k.detach().clamp(0, 1), h_k.detach()
        return x_k.detach(), h_k.detach()

    def refine_q_hmc(x_init, h_init, n_steps, sgld_step, beta=.5):
        x_k = torch.clone(x_init).requires_grad_()
        h_k = torch.clone(h_init).requires_grad_()
        v_x = torch.zeros_like(x_k)
        v_h = torch.zeros_like(h_k)
        sgld_sigma_tilde = (2 * sgld_step) ** .5
        sgld_sigma = (2 * beta * sgld_step) ** .5
        for k in range(n_steps):
            logp = logp_fn(x_k)[:, 0]
            logq = logq_unnorm(x_k, h_k)
            g_h = torch.autograd.grad(logq.sum(), [h_k], retain_graph=True)[0]

            # sample h tilde ~ q(h|x_k)
            h_tilde = h_k + g_h * sgld_step + torch.randn_like(h_k) * sgld_sigma_tilde
            h_tilde = h_tilde.detach()

            logq_tilde = logq_unnorm(x_k, h_tilde)
            logq = logq_unnorm(x_k, h_k)
            g_x = torch.autograd.grad(logp.sum() + logq.sum() - logq_tilde.sum(), [x_k], retain_graph=True)[0]
            # x update
            v_x = (v_x * (1 - beta) + sgld_step * g_x + torch.randn_like(x_k) * sgld_sigma).detach()
            x_k = (x_k + v_x).detach().requires_grad_().clamp(0, 1)

            # h update
            logq = logq_unnorm(x_k, h_k)
            g_h = torch.autograd.grad(logq.sum(), [h_k], retain_graph=True)[0]

            v_h = (v_h * (1 - beta) + sgld_step * g_h + torch.randn_like(h_k) * sgld_sigma).detach()
            h_k = (h_k + v_h).detach().requires_grad_()

        return x_k.detach().clamp(0, 1), h_k.detach()

    g.train()
    g.to(device)
    logp_net.to(device)

    itr = 0
    for epoch in range(args.n_epochs):
        for x_d, _ in train_loader:
            optimizer.zero_grad()
            if args.dataset in TOY_DSETS:
                x_d = toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)
                x_d = torch.from_numpy(x_d).float().to(device)
            else:
                x_d = x_d.to(device)


            x_init, h_init = sample_q(args.batch_size)
            if args.hmc:
                x, h = refine_q(x_init, h_init, args.n_steps, args.sgld_step)
            else:
                x, h = refine_q_hmc(x_init, h_init, args.n_steps, args.sgld_step)

            ld = logp_fn(x_d)[:, 0]
            lm = logp_fn(x.detach())[:, 0]
            li = logp_fn(x_init.detach())[:, 0]
            logp_obj = (ld - lm).mean()
            logq_obj = logq_unnorm(x.detach(), h.detach()).mean()

            loss = -(logp_obj + 3 * logq_obj) + args.p_control * (ld**2).mean()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("({}) | log p obj = {:.4f}, log q obj = {:.4f}, sigma = {:.4f} | log p(x_d) = {:.4f}, log p(x_m) = {:.4f}, log p(x_i) = {:.4f}".format(
                    itr, logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(), ld.mean().item(), lm.mean().item(), li.mean().item()))

            if itr % args.viz_every == 0:
                if args.dataset in TOY_DSETS:
                    plt.clf()
                    xm = x.cpu().numpy()
                    xn = x_d.cpu().numpy()
                    xi = x_init.detach().cpu().numpy()
                    ax = plt.subplot(1, 5, 1, aspect="equal", title='refined')
                    ax.scatter(xm[:, 0], xm[:, 1], s=1)

                    ax = plt.subplot(1, 5, 2, aspect="equal", title='initial')
                    ax.scatter(xi[:, 0], xi[:, 1], s=1)

                    ax = plt.subplot(1, 5, 3, aspect="equal", title='data')
                    ax.scatter(xn[:, 0], xn[:, 1], s=1)

                    ax = plt.subplot(1, 5, 4, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_fn, ax, low=x_d.min().item(), high=x_d.max().item())
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)

                    ax = plt.subplot(1, 5, 5, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_fn, ax, low=x_d.min().item(), high=x_d.max().item(), exp=False)
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)
                else:
                    plot("{}/init_{}.png".format(args.save_dir, itr), x_init.view(x_init.size(0), *args.data_size))
                    plot("{}/ref_{}.png".format(args.save_dir, itr), x.view(x.size(0), *args.data_size))
                    plot("{}/data_{}.png".format(args.save_dir, itr), x_d.view(x_d.size(0), *args.data_size))

            itr += 1



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
                                    # transform=transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)]),
                                    transform=transforms.Compose([transforms.ToTensor(),
                                                                  lambda x: (((255. * x) + torch.rand_like(x)) / 256.).view(-1)]),
                                    download=True)
        te_dataset = datasets.MNIST("./data", train=False,
                                    transform=transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)]),
                                    download=True)
        tr_dload = DataLoader(tr_dataset, args.batch_size, True, drop_last=True)
        te_dload = DataLoader(te_dataset, args.batch_size, False)
        return tr_dload, te_dload
    else:
        raise NotImplementedError




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="circles")#, choices=["mnist", "moons", "circles"])
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
    parser.add_argument("--p_control", type=float, default=0.0)
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=10,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--sgld_step", type=float, default=.01)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='/tmp/')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=20, help="Iterations between print")
    parser.add_argument("--viz_every", type=int, default=200, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--semi-supervised", type=bool, default=False)
    parser.add_argument("--vat", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--hmc", action="store_true", help="Run VAT instead of JEM")

    args = parser.parse_args()
    if args.dataset in TOY_DSETS:
        args.data_dim = 2
    elif args.dataset == "mnist":
        args.data_dim = 784
        args.data_size = (1, 28, 28)

    main(args)
