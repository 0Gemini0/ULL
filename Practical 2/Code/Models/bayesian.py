"""
Bayesian skipgram model in PyTorch.
"""
import torch
from torch import nn
from torch.distributions import Normal
import torch.functional as F


class Bayesian(nn.Module):
    """
    Bayesian SkipGram model in PyTorch.
    The Inference model encodes center words into a posterior representation given the context words.
    Training is done with Variational Inference. The prior over words will contain the final embeddings.
    Inputs:
    * v_dim (int): size of the vocabulary.
    * d_dim (int): size of the embeddings.
    * h_dim (int): size of hidden layers.
    """

    def __init__(self, v_dim, d_dim, h_dim):
        super().__init__()

        ## TODO: layers
        self.prior = Prior(v_dim, d_dim)
        self.posterior = Posterior(v_dim, d_dim, h_dim)
        self.standard_normal = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

    def forward(self, center, context):
        """The model samples an encoding from the posterior."""
        mu, sigma = self.posterior(center, context)
        z = self.sample(mu, sigma)

    def _kl_divergence(self, mu_1, sigma_1, mu_2, sigma_2):
        """
        Computes the KL-divergence between two Gaussian distributions with diagonal covariance given their mean and covariance.
        """
        return -1. / self.batch_size * 0.5 * torch.sum(torch.log(sigma_2) - torch.log(sigma_1)
                                                       + sigma_1 / sigma_2 + (mu_2 - mu_1) ** 2 / sigma_2 - 1)

    def sample(self, mu, sigma):
        """Reparameterized sampling from a Gaussian density."""
        return mu + sigma * self.standard_normal.sample_n(mu.shape[0])


class Prior(nn.Module):
    """Small feedforward module that outputs the (prior) mean and sigma of a Gaussian."""

    def __init__(self, v_dim, d_dim):
        super().__init__()

        # Embeddings for Mu and Sigma, only dependent on the words p(z|w); the Gaussians are spherical
        self.mu = nn.Embedding(v_dim, e_dim, sparse=True)
        self.sigma = nn.Embedding(v_dim, 1, sparse=True)

    def forward(self, x):
        """Embed word x into a d-dimensional mu and (diagonal) (positive) sigma vector."""
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))

        return mu, sigma


class Posterior(nn.Module):
    """Small feedforward module that outputs the (posterior) mean and sigma of a Gaussian."""

    def __init__(self, v_dim, d_dim, h_dim):
        super().__init__()

        self.h_dim = h_dim

        # The first layer is just a random embedding of the center and context words
        self.embedding = nn.Embedding(v_dim, d_dim)

        # Mean and sigma of the posterior Gaussians, which are again spherical
        self.linear = nn.Linear(d_dim * 2, h_dim)
        self.mean = nn.Linear(h_dim, d_dim)
        self.sigma = nn.Linear(h_dim, 1)

    def forward(self, center, contexts):
        """Input is assumed to be a sequence of sequences of torch tensors with size [batch_size x dim]."""
        center = self.embedding(center)
        pairs = []
        lengths = []

        for i, context in enumerate(contexts):
            context = self.embedding(context)

            # We keep track of the batch_size of the contexts, as not all central words in the batch may have the same
            # number of context words
            lengths.append(context.shape[0])

            # We append each center word embedding to its context word embedding, given that it has a context word in
            # position i
            pairs.append(torch.cat([center[:lengths[i], :], context], dim=1))

        # Stack the concatenated pairs into a single input of the linear layer
        x = torch.cat(pairs, dim=0)
        h = F.relu(self.linear(x))

        # Split and sum into a single hidden layer
        h_list = torch.split(h, lengths, dim=0)
        h_sum = torch.zeros([center.shape[0], self.h_dim])
        for h, length in zip(h_list, lengths):
            h_sum[:length, :] += h

        # Compute mean and sigma
        mean = self.mean(h_sum)
        sigma = F.softplus(self.sigma(h_dim))  # We don't use log_sigma as in the paper for stability reasons

        return mean, sigma
