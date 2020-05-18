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
TOY_DSETS = ("moons", "circles", "8gaussians", "pinwheel", "2spirals", "checkerboard", "rings", "swissroll")

def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)


def main(args):
    utils.makedirs(args.save_dir)
    logp_net, g = get_models(args)

    e_optimizer = torch.optim.Adam(logp_net.parameters(), lr=args.lr, betas=[0.5, .999], weight_decay=args.weight_decay)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=args.lr / 1, betas=[0.5, .999], weight_decay=args.weight_decay)

    train_loader, test_loader, plot = get_data(args)



    def sample_q(n):
        h = torch.randn((n, args.noise_dim)).to(device)
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
    stepsize_x = 1. / args.data_dim
    es = []
    for epoch in range(args.n_epochs):
        for x_d, _ in train_loader:
            if args.dataset in TOY_DSETS:
                x_d = toy_data.inf_train_gen(args.dataset, batch_size=args.batch_size)
                x_d = torch.from_numpy(x_d).float().to(device)
            else:
                x_d = x_d.to(device)

            # sample from q(x, h)
            x_g, h_g = sample_q(args.batch_size)
            x_gc = x_g.clone()
            if args.refine:
                x_g_ref, acceptRate_x, stepsize_x = hmc.get_ebm_samples(logp_net, x_g.detach(),
                                                                        burn_in=2, num_samples_posterior=1,
                                                                        leapfrog_steps=5,
                                                                        stepsize=stepsize_x, flag_adapt=1,
                                                                        hmc_learning_rate=.02, hmc_opt_accept=.67)
            elif args.refine_latent:
                h_g_ref, eps_ref, acceptRate_x, stepsize_x = hmc.get_ebm_latent_samples(logp_net, g.generator,
                                                                                        h_g.detach(), torch.randn_like(x_g).detach(),
                                                                                        g.logsigma.exp().detach(),
                                                                                        burn_in=2, num_samples_posterior=1,
                                                                                        leapfrog_steps=5,
                                                                                        stepsize=stepsize_x, flag_adapt=1,
                                                                                        hmc_learning_rate=.02, hmc_opt_accept=.67)
                x_g_ref = g.generator(h_g_ref) + eps_ref * g.logsigma.exp()
            else:
                x_g_ref = x_g
            # ebm obj
            ld = logp_net(x_d).squeeze()
            lg_detach = logp_net(x_g_ref.detach()).squeeze()
            logp_obj = (ld - lg_detach).mean()

            if args.stagger:
                if lg_detach.mean() > ld.mean() - 2 * ld.std() or itr < 100:
                    e_loss = -logp_obj + (ld ** 2).mean() * args.p_control
                    e_optimizer.zero_grad()
                    e_loss.backward()
                    e_optimizer.step()
                    #print('e')
                    if itr < 100:
                        es.append(1)
                    else:
                        es = es[1:] + [1]
                else:
                    #print('no-e', lg_detach.mean(), ld.mean(), ld.std())
                    es = es[1:] + [0]
                if itr % args.print_every == 0:
                    print('e frac', np.mean(es))
            else:
                e_loss = -logp_obj + (ld ** 2).mean() * args.p_control
                e_optimizer.zero_grad()
                e_loss.backward()
                e_optimizer.step()

            # gen obj
            if args.mode == "reverse_kl":
                x_g, h_g = sample_q(args.batch_size)

                lg = logp_net(x_g).squeeze()
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
                logq_obj = lg.mean() + g_error_entropy

            elif args.mode == "kl":
                h_given_x, acceptRate, stepsize = hmc.get_gen_posterior_samples(
                    g.generator, x_g_ref.detach(), h_g.clone(), g.logsigma.exp().detach(), burn_in=2,
                    num_samples_posterior=1, leapfrog_steps=5, stepsize=stepsize, flag_adapt=1,
                    hmc_learning_rate=.02, hmc_opt_accept=.67
                )
                logq_obj = logq_joint(x_g_ref.detach(), h_given_x.detach()).mean()
                g_error_entropy = torch.zeros_like(logq_obj)
            else:
                raise ValueError

            g_loss = -logq_obj
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g.logsigma.data.clamp_(np.log(.01), np.log(.3))

            if itr % args.print_every == 0:
                delta = (x_gc - x_g_ref).flatten(start_dim=1).norm(dim=1).mean()
                print("({}) | log p obj = {:.4f}, log q obj = {:.4f}, sigma = {:.4f} | "
                      "log p(x_d) = {:.4f}, log p(x_m) = {:.4f}, ent = {:.4f} | "
                      "stepsize = {:.4f}, stepsize_x = {} | delta = {:.4f}".format(
                    itr, logp_obj.item(), logq_obj.item(), g.logsigma.exp().item(),
                    ld.mean().item(), lg_detach.mean().item(), g_error_entropy.item(),
                    stepsize, stepsize_x, delta.item()))

            if itr % args.viz_every == 0:
                if args.dataset in TOY_DSETS:
                    plt.clf()
                    xg = x_gc.detach().cpu().numpy()
                    xgr = x_g_ref.detach().cpu().numpy()
                    xd = x_d.cpu().numpy()

                    ax = plt.subplot(1, 5, 1, aspect="equal", title='init')
                    ax.scatter(xg[:, 0], xg[:, 1], s=1)

                    ax = plt.subplot(1, 5, 2, aspect="equal", title='refined')
                    ax.scatter(xgr[:, 0], xgr[:, 1], s=1)

                    ax = plt.subplot(1, 5, 3, aspect="equal", title='data')
                    ax.scatter(xd[:, 0], xd[:, 1], s=1)

                    ax = plt.subplot(1, 5, 4, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_net, ax, low=x_d.min().item(), high=x_d.max().item())
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)

                    ax = plt.subplot(1, 5, 5, aspect="equal")
                    logp_net.cpu()
                    utils.plt_flow_density(logp_net, ax, low=x_d.min().item(), high=x_d.max().item(), exp=False)
                    plt.savefig("/{}/{}.png".format(args.save_dir, itr))
                    logp_net.to(device)
                else:
                    plot("{}/{}_init.png".format(args.save_dir, itr), x_gc.view(x_g.size(0), *args.data_size))
                    plot("{}/{}_ref.png".format(args.save_dir, itr), x_g_ref.view(x_g.size(0), *args.data_size))

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
                self.logsigma = nn.Parameter((torch.zeros(1, ) + 1.).log())
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
                self.logsigma = nn.Parameter((-torch.ones(1, )))

    elif args.dataset == "svhn":
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
        logp_net = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.utils.weight_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.utils.weight_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.utils.weight_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.utils.weight_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
            nn.Conv2d(512, 1, 2, 1, 0, bias=False),
            #nn.Sigmoid()
        )



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
                self.generator = Generator()
                self.logsigma = nn.Parameter((-torch.ones(1, )))

    return logp_net, G()




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    #cifar
    parser.add_argument("--dataset", type=str, default="circles")#, choices=["mnist", "moons", "circles"])
    parser.add_argument("--mode", type=str, default="reverse_kl")
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
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--e_iters", type=int, default=1)
    parser.add_argument("--g_iters", type=int, default=1)
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
    parser.add_argument("--stagger", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--hmc", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--refine", action="store_true", help="Run VAT instead of JEM")
    parser.add_argument("--refine_latent", action="store_true", help="Run VAT instead of JEM")

    args = parser.parse_args()
    if args.dataset in TOY_DSETS:
        args.data_dim = 2
    elif args.dataset == "mnist":
        args.data_dim = 784
        args.data_size = (1, 28, 28)
    elif args.dataset == "svhn":
        args.data_dim = 32 * 32 * 3
        args.data_size = (3, 32, 32)

    main(args)
