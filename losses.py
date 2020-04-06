import torch as t
import torch.nn as nn


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= t.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    # Adapted from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py

    def __init__(self, xi=10.0, eps=2.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with t.no_grad():
            pred = t.nn.functional.softmax(model.classify(x), dim=1)

        # prepare random unit tensor
        d = t.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat = model.classify(x + self.xi * d)
            logp_hat = t.nn.functional.log_softmax(pred_hat, dim=1)
            adv_distance = t.nn.functional.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            model.zero_grad()

        # calc LDS
        r_adv = d * self.eps
        pred_hat = model.classify(x + r_adv)
        logp_hat = t.nn.functional.log_softmax(pred_hat, dim=1)
        lds = t.nn.functional.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


class LDSLoss(nn.Module):
    # Adapted from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py

    def __init__(self, n_steps):
        # n_steps = args.label_prop_n_steps
        super(LDSLoss, self).__init__()
        self.n_steps = n_steps

    def forward(self, model, x, sample_q, seed_batch):
        with t.no_grad():
            pred = t.nn.functional.softmax(model.classify(x), dim=1)

        # get a sample with a certain number of steps
        # args.label_prop_n_steps
        samples = sample_q(model, replay_buffer=[], y=pred.argmax(dim=1), n_steps=self.n_steps, seed_batch=seed_batch)

        # calc LDS between prediction on sample and prediction on original
        pred_hat = model.classify(samples)
        logp_hat = t.nn.functional.log_softmax(pred_hat, dim=1)
        lds = t.nn.functional.kl_div(logp_hat, pred, reduction='batchmean')

        return lds



def sliced_score_matching(f, samples, n_particles=1):
    # Adapted from: https://github.com/ermongroup/sliced_score_matching/blob/master/losses/sliced_sm.py
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = t.randn_like(dup_samples)
    vectors = vectors / t.norm(vectors, dim=-1, keepdim=True)

    logits = f.classify(dup_samples)
    logp = logits.logsumexp(1).sum()
    grad1 = t.autograd.grad(logp, dup_samples, create_graph=True)[0]
    gradv = t.sum(grad1 * vectors)
    loss1 = t.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    grad2 = t.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = t.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss1 + loss2
    return loss.mean()


def sliced_score_matching_vr(f, samples, n_particles=1):
    # Adapted from: https://github.com/ermongroup/sliced_score_matching/blob/master/losses/sliced_sm.py
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = t.randn_like(dup_samples)

    logits = f.classify(dup_samples)
    logp = logits.logsumexp(1).sum()
    grad1 = t.autograd.grad(logp, dup_samples, create_graph=True)[0]
    loss1 = t.sum(grad1 * grad1, dim=-1) / 2.
    gradv = t.sum(grad1 * vectors)
    grad2 = t.autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = t.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean()

