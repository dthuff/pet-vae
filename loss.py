import torch
import torch.nn as nn


class KLDivergence(nn.Module):
    "KL divergence between the estimated normal distribution and a prior distribution"

    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, z_mean, z_log_sigma):
        z_log_var = z_log_sigma * 2
        # See (2) from https://statproofbook.github.io/P/norm-kl.html.
        # For Q = N(0,1), KL(P|Q) is the following:
        return 0.5 * ((z_mean ** 2 + z_log_var.exp() - z_log_var - 1).sum())


class L2Loss(nn.Module):
    "L2 norm between predictions x and label y"

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x, y):
        return ((x - y) ** 2).mean()


class L1Loss(nn.Module):
    "Measuring the `Euclidian distance` between prediction and ground truh using `L1 Norm`"

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
        return ((x - y).abs()).mean()
