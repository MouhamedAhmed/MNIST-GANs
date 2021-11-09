import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNorm(nn.Module):
    def __init__(self, module):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.u = None

    def forward(self, *args):
        # weights
        weights = self.module.weight

        # add u to model params if not exists
        if self.u is None:
            self.u = F.normalize(weights.data.new(weights.data.shape[0]).normal_(0, 1), dim = 0)

        weights_view_data = weights.view(weights.data.shape[0],-1).data
        # v = Normalize(weights.T * u)
        v = F.normalize(torch.mv(torch.t(weights_view_data), self.u), dim = 0)
        # u = Normalize(weights * v)
        self.u = F.normalize(torch.mv(weights_view_data, v), dim = 0)

        # update weights, weights = weights / (u * weights * v)
        weights.data = weights.data / torch.dot(self.u, torch.mv(weights_view_data, v))

        return self.module.forward(*args)