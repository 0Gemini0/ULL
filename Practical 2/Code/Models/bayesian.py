"""
Bayesian skipgram model in PyTorch.
"""
import torch
from torch import nn
import torch.functional as F


class Bayesian(nn.Module):
    """Bayesian SkipGram model in PyTorch."""

    def __init__(self, v_dim, e_dim, h_dim):
        super(Bayesian, self).__init__()

        ## TODO: layers
        self.encoder = Encoder(e_dim * 2, h_dim)

    def forward(self):
        ## TODO: forward

    def sample(self):
        ## TODO: sample


class Encoder(nn.Module):
    """Small feedforward module that outputs the mean and sigma of a Gaussian."""

    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()

        # Model structure
        self.linear = nn.Linear(in_dim, out_dim)
        self.mean = nn.Linear(out_dim, 1)
        self.sigma = nn.Linear(out_dim, 1)

    def forward(self, inp):
        """input is assumed to be a sequence of sequences of torch tensors with size [batch_size x dim]"""
        h = []
        for pair in inp:
            # Stack inputs along the second dimension
            x = torch.cat(pair, dim=1)

            # Forward pass
            h.append(F.relu(self.linear(x)))

        # Sum into single hidden layer
        h = h.sum()

        # Compute mean and sigma
        mean = self.mean(h)
        log_sigma = self.sigma(h)

        return mean, log_sigma
