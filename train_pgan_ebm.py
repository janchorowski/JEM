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
TOY_DSETS = ("moons", "circles", "8gaussians", "pinwheel", "2spirals", "checkerboard", "rings", "swissroll")

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)


def _helper(netG, x_tilde, eps, sigma):
    eps = eps.clone().detach().requires_grad_(True)
    with torch.no_grad():
        G_eps = netG(eps)
    bsz = eps.size(0)
    log_prob_eps = (eps ** 2).view(bsz, -1).sum(1).view(-1, 1)
    log_prob_x = (x_tilde - G_eps)**2 / sigma**2
    log_prob_x = log_prob_x.view(bsz, -1)
    log_prob_x = torch.sum(log_prob_x, dim=1).view(-1, 1)
    logjoint_vect = -0.5 * (log_prob_eps + log_prob_x)
    logjoint_vect = logjoint_vect.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = eps.grad
    return logjoint_vect, logjoint, grad_logjoint


def get_samples(netG, x_tilde, eps_init, sigma, burn_in, num_samples_posterior,
            leapfrog_steps, stepsize, flag_adapt, hmc_learning_rate, hmc_opt_accept):
    device = eps_init.device
    bsz, eps_dim = eps_init.size(0), eps_init.size(1)
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz*num_samples_posterior, eps_dim).to(device)
    current_eps = eps_init
    cnt = 0
    for i in range(n_steps):
        eps = current_eps
        p = torch.randn_like(current_eps)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, current_eps, sigma)
        current_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            eps = eps + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, eps, sigma)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _helper(netG, x_tilde, eps, sigma)
        proposed_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p**2).sum(dim=1)
        current_K = current_K.view(-1, 1) ## should be size of B x 1
        proposed_K = 0.5 * (p**2).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1) ## should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K))
        accept = accept.float().squeeze() ## should be B x 1
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
            stepsize = stepsize + hmc_learning_rate * (accept.float().mean() - hmc_opt_accept) * stepsize
        else:
            samples[cnt*bsz : (cnt+1)*bsz, :] = current_eps.squeeze()
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize


def main(args):
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
                self.logsigma = nn.Parameter((torch.zeros(1,) + 1.).log())
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
                self.logsigma = nn.Parameter((-torch.ones(1,)))


    g = G()


    e_optimizer = torch.optim.Adam(logp_net.parameters(), lr=args.lr, betas=[0.5, .999], weight_decay=args.weight_decay)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=args.lr / 1, betas=[0.5, .999], weight_decay=args.weight_decay)

    train_loader, test_loader = get_data(args)

    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: torchvision.utils.save_image(torch.clamp(x, 0, 1), p, normalize=False, nrow=sqrt(x.size(0)))

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
                x_d = toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)
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
                print("({}) | log p obj = {:.4f}, log q obj = {:.4f}, sigma = {:.4f} | "
                      "log p(x_d) = {:.4f}, log p(x_m) = {:.4f}, ent = {:.4f} | "
                      "stepsize = {:.4f}".format(
                    itr, logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(),
                    ld.mean().item(), lg.mean().item(), g_error_entropy.item(), stepsize.item()))

            if itr % args.viz_every == 0:
                if args.dataset in TOY_DSETS:
                    plt.clf()
                    xg = x_g.detach().cpu().numpy()
                    xd = x_d.cpu().numpy()
                    ax = plt.subplot(1, 4, 1, aspect="equal", title='refined')
                    ax.scatter(xg[:, 0], xg[:, 1], s=1)

                    ax = plt.subplot(1, 4, 2, aspect="equal", title='data')
                    ax.scatter(xd[:, 0], xd[:, 1], s=1)

                    ax = plt.subplot(1, 4, 3, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_fn, ax, low=x_d.min().item(), high=x_d.max().item())
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)

                    ax = plt.subplot(1, 4, 4, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_fn, ax, low=x_d.min().item(), high=x_d.max().item(), exp=False)
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)
                else:
                    plot("{}/ref_{}.png".format(args.save_dir, itr), x_g.view(x_g.size(0), *args.data_size))
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
    parser.add_argument("--dataset", type=str, default="rings")#, choices=["mnist", "moons", "circles"])
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
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--n_epochs", type=int, default=1000)
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
    parser.add_argument("--save_dir", type=str, default='/tmp/pgan')
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
