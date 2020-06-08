import argparse
import torch
import torch.distributions as distributions
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import utils
import networks
from sklearn import datasets
from matplotlib.colors import ListedColormap
import data_utils
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')





def decision_boundary(net, X):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    xxt = torch.from_numpy(xx.ravel()).float()
    yyt = torch.from_numpy(yy.ravel()).float()
    xxyy = torch.cat([xxt[:, None], yyt[:, None]], dim=1)
    logits = net(xxyy)
    Z = logits.argmax(1)
    plt.pcolormesh(xx, yy, Z.numpy().reshape(xx.shape), cmap=ListedColormap(['r', 'b']), alpha=.1)




def main(args):
    utils.makedirs(args.save)
    net = networks.SmallMLP(2, 2, n_hid=args.hid)

    if args.dataset == "moons":
        Xf, Y = datasets.make_moons(1000, noise=.1)
        Xfte, Yte = datasets.make_moons(1000, noise=.1)
        Xoh, Xohte = [], []
    elif args.dataset == "circles":
        Xf, Y = datasets.make_circles(1000, noise=.03)
        Xfte, Yte = datasets.make_circles(1000, noise=.03)
        Xoh, Xohte = [], []
    elif args.dataset == "adult":
        with open("data/adult/adult.data", 'r') as f:
            Xf, Xoh, Y = data_utils.load_adult()
        with open("data/adult/adult.test", 'r') as f:
            Xfte, Xohte, Yte = data_utils.load_adult()

    else:
        raise NotImplementedError

    Xf = Xf.astype(np.float32)
    Xfl, Xohl, Yl = [], [], []
    if args.n_labels_per_class != -1:
        Xfl.extend(Xf[Y == 0][:args.n_labels_per_class])
        Xfl.extend(Xf[Y == 1][:args.n_labels_per_class])
        Yl.extend([0] * args.n_labels_per_class)
        Yl.extend([1] * args.n_labels_per_class)
        if Xoh is not None:
            Xohl.extend(Xf[Y == 0][:args.n_labels_per_class])
            Xohl.extend(Xf[Y == 1][:args.n_labels_per_class])
    else:
        Xfl, Xohl, Yl = Xf, Xoh, Y

    def plot_data(fname="data.png"):
        plt.clf()
        decision_boundary(net, Xf)
        plt.scatter(Xf[:, 0], Xf[:, 1], c='grey')
        plt.scatter(Xfl[:args.n_labels_per_class, 0], Xfl[:args.n_labels_per_class, 1], c='r')
        plt.scatter(Xfl[args.n_labels_per_class:, 0], Xfl[args.n_labels_per_class:, 1], c='b')
        plt.savefig("{}/{}".format(args.save, fname))

    optim = torch.optim.Adam(params=net.parameters(), lr=args.lr)

    xl = torch.from_numpy(Xl).to(device)
    yl = torch.from_numpy(np.array(Yl)).to(device)
    x_te, y_te = torch.from_numpy(Xte).float(), torch.from_numpy(Yte)
    inds = list(range(X.shape[0]))
    for i in range(args.n_iters):
        batch_inds = np.random.choice(inds, args.batch_size, replace=False)
        x = X[batch_inds]
        x = torch.from_numpy(x).to(device).requires_grad_()

        logits = net(xl)
        clf_loss = nn.CrossEntropyLoss(reduction='none')(logits, yl).mean()

        logits_u = net(x)
        logpx_plus_Z = logits_u.logsumexp(1)
        sp = utils.keep_grad(logpx_plus_Z.sum(), x)
        e = torch.randn_like(sp)
        eH = utils.keep_grad(sp, x, grad_outputs=e)
        trH = (eH * e).sum(-1)

        sm_loss = trH + .5 * (sp ** 2).sum(-1)
        sm_loss = sm_loss.mean()

        loss = (1 - args.sm_lam) * clf_loss + args.sm_lam * sm_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 100 == 0:
            if args.dataset in ("rings", "moons"):
                plot_data("data_{}.png".format(i))
            te_logits = net(x_te.float())
            te_preds = torch.argmax(te_logits, 1)
            te_acc = (te_preds == y_te).float().mean()
            print("Iter {}: Clf Loss = {}, SM Loss = {} | Test Accuracy = {}".format(i,
                                                                                     clf_loss.item(), sm_loss.item(),
                                                                                     te_acc.item()))





if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    # logging + evaluation
    parser.add_argument("--save", type=str, default='.')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--hid", type=int, default=100)
    parser.add_argument("--n_labels_per_class", type=int, default=3)
    parser.add_argument("--n_iters", type=int, default=10000)
    parser.add_argument("--sm_lam", type=float, default=.8)
    parser.add_argument("--dist", type=str, default="gaussian")
    parser.add_argument("--posterior", type=str, default="gaussian-1")
    parser.add_argument("--dataset", type=str, default="moons")
    parser.add_argument("--std_sample", type=str, default="log", choices=["log", "linear"])
    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[])
    parser.add_argument("--decay_rate", type=float, default=.1)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=500)
    # regularization
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--network", type=str, default="mlp", choices=["mlp", "resnet"])
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sample_every", type=int, default=100)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true")
    parser.add_argument("--form", type=str, default="critic")
    parser.add_argument("--direct_loss", action="store_true")
    parser.add_argument("--logit", action="store_true")

    args = parser.parse_args()
    main(args)
