import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def brute_force_jac(x_g, h_g):
    jac = torch.zeros((x_g.size(0), x_g.size(1), h_g.size(1)))
    for d in range(x_g.size(1)):
        j = torch.autograd.grad(x_g[:, d].sum(), h_g, retain_graph=True)[0]
        jac[:, d, :] = j
    return jac

def normalize(x, eps=1e-6):
    xs = x.view(x.size(0), -1)
    xn = xs / (xs.norm(dim=1, keepdim=True) + eps)
    return xn.view(*x.size())

def find_extreme_singular_vectors(net, x, niters=100, v_norm=.01, mode="min", lr=1., log=False):
    v = torch.randn_like(x)
    v = nn.Parameter(v)
    optim = torch.optim.Adam([v], lr)
    for i in range(niters):
        optim.zero_grad()
        gx = net(x)
        vn = v_norm * normalize(v)

        xpv = x + vn
        gxpv = net(xpv)

        gdiff = (gx - gxpv).flatten(start_dim=1).norm(dim=1)
        target = (gdiff / v_norm)
        if log:
            target = target.log()
        if mode == "max":
            target = -target

        target.sum().backward()
        optim.step()
    return normalize(v).detach(), target


def log_sigular_values_sum_bound(net, x, v, v_norm=.01):
    v = normalize(v) * v_norm
    gx = net(x)
    gxpv = net(x + v)
    diff_norm = (gx - gxpv).view(x.size(0), -1).norm(dim=1)
    log_diff_norm = diff_norm.log()
    log_min_sv = log_diff_norm - np.log(v_norm)
    return log_min_sv * np.prod(x.size()[1:])

def main():
    din = 100
    dout = 784
    dh = 1000
    net = nn.Sequential(
        nn.Linear(din, dh),
        nn.ReLU(),
        nn.Linear(dh, dh),
        nn.ReLU(),
        nn.Linear(dh, dout)
    )
    x = torch.randn((2, din)).requires_grad_()
    y = net(x)

    j = brute_force_jac(y, x)#[0]

    u, s, v = torch.svd(j, compute_uv=False)
    #a = j.t()
    print(s[:, 0], s[:, -1])
    print(s[:, -1].log() * din)
    #print(s)

    u = torch.randn((1, din))
    u = u / u.norm()
    #print(u.norm())

    #out = u @ a
    #print(out.norm())
    #u_min = nn.Parameter(u)
    #u_max = nn.Parameter(u.clone())
    #optim = torch.optim.Adam([u_min, u_max], .1)

    # for i in range(100):
    #     optim.zero_grad()
    #     umi = u_min / (u_min.norm() + 1e-6)
    #     obj_max = (umi @ a).norm()
    #
    #     uma = u_max / (u_max.norm() + 1e-6)
    #     obj_min = -(uma @ a).norm()
    #     if i % 1 == 0:
    #         print(i, -obj_min, obj_max)
    #     loss = -obj_max - obj_min
    #     loss.backward()
    #     optim.step()


    # v_norm = .01
    # v = torch.randn_like(x)
    # v_min = nn.Parameter(v)
    # v_max = nn.Parameter(v.clone())
    # optim = torch.optim.Adam([v_min, v_max], 1.)
    # for i in range(100):
    #     optim.zero_grad()
    #     gx = net(x)
    #     vn = v_norm * v_min / (v_min.norm(dim=1) + 1e-6)
    #
    #     xpv = x + vn
    #     gxpv = net(xpv)
    #
    #     gdiff = (gx - gxpv).norm(dim=1)
    #     target_min = (gdiff / v_norm).log()
    #
    #     vn = v_norm * v_max / (v_max.norm(dim=1) + 1e-6)
    #
    #     xpv = x + vn
    #     gxpv = net(xpv)
    #
    #     gdiff = (gx - gxpv).norm(dim=1)
    #     target_max = -(gdiff / v_norm).log()
    #
    #     loss = target_min + target_max
    #     loss.backward()
    #     optim.step()
    #     print(target_min, -target_max)

    v, t = find_extreme_singular_vectors(lambda y: net(y).view(y.size(0), 1, 28, 28), x, niters=10)
    # print(t.size(), v.size())
    #
    # print(t)
    # print(v.norm(dim=1))
    ssv = log_sigular_values_sum_bound(lambda y: net(y).view(y.size(0), 1, 28, 28), x, v)
    #print(t.log() * din)
    # print(ssv)
    # print(s[:, -1].log() * din)
    # print()
    print(ssv - s[:, -1].log() * din, "lr = 1.")

    v, t = find_extreme_singular_vectors(lambda y: net(y).view(y.size(0), 1, 28, 28), x, niters=10, lr=.1)
    ssv = log_sigular_values_sum_bound(lambda y: net(y).view(y.size(0), 1, 28, 28), x, v)
    # print(t.log() * din)
    # print(ssv)
    # print(s[:, -1].log() * din)
    # print()
    print(ssv - s[:, -1].log() * din, "lr = .1")

    v, t = find_extreme_singular_vectors(lambda y: net(y).view(y.size(0), 1, 28, 28), x, niters=10, lr=10)
    ssv = log_sigular_values_sum_bound(lambda y: net(y).view(y.size(0), 1, 28, 28), x, v)
    # print(t.log() * din)
    # print(ssv)
    # print(s[:, -1].log() * din)
    # print()
    print(ssv - s[:, -1].log() * din, "lr = 10")





    v, t = find_extreme_singular_vectors(lambda y: net(y).view(y.size(0), 1, 28, 28), x, niters=10, log=True)
    # print(t.size(), v.size())
    #
    # print(t)
    # print(v.norm(dim=1))
    ssv = log_sigular_values_sum_bound(lambda y: net(y).view(y.size(0), 1, 28, 28), x, v)
    # print(t.log() * din)
    # print(ssv)
    # print(s[:, -1].log() * din)
    # print()
    print(ssv - s[:, -1].log() * din, "lr = 1. (log)")

    v, t = find_extreme_singular_vectors(lambda y: net(y).view(y.size(0), 1, 28, 28), x, niters=10, lr=.1, log=True)
    ssv = log_sigular_values_sum_bound(lambda y: net(y).view(y.size(0), 1, 28, 28), x, v)
    # print(t.log() * din)
    #print(ssv)
    #print(s[:, -1].log() * din)
    #print()
    print(ssv - s[:, -1].log() * din, "lr = .1 (log)")

    v, t = find_extreme_singular_vectors(lambda y: net(y).view(y.size(0), 1, 28, 28), x, niters=10, lr=10, log=True)
    ssv = log_sigular_values_sum_bound(lambda y: net(y).view(y.size(0), 1, 28, 28), x, v)
    # print(t.log() * din)
    #print(ssv)
    #print(s[:, -1].log() * din)
    #print()
    print(ssv - s[:, -1].log() * din, "lr = 10 (log)")

    #1/0



if __name__ == "__main__":
    main()

