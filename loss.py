import torch
from torch import nn
import torch.nn.functional as F


class GramMatrix(nn.Module):
    """Return gram matrix of the input y

    Arguments:
        y {[matrix]} -- shape (n_b, n_C, n_H, n_W)

    Returns:
        Gram matrix of A, of shape (n_C, n_C)
    """
    def forward(self, x):
        b, c, h, w = x.size()
        F = x.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # Manually compute gradients => detach
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # Manually compute gradients => detach
        self.target = GramMatrix()(target_feature).detach()

    def forward(self, x):
        G = GramMatrix()(x)
        self.loss = F.mse_loss(G, self.target)
        return x

