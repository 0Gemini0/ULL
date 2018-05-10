"""
Bayesian skipgram model in PyTorch.
"""
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F


class Bayesian(nn.Module):
    """
    Bayesian SkipGram model in PyTorch.
    The Inference model encodes center words into a posterior representation given the context words.
    Training is done with Variational Inference. The prior over words will contain the final embeddings.
    Inputs:
    * v_dim (int): size of the vocabulary.
    * d_dim (int): size of the embeddings.
    * h_dim (int): size of hidden layers.
    * pad_index (int): index of <pad> token.
    """

    def __init__(self, v_dim, d_dim, h_dim, pad_index):
        super().__init__()

        # Layers
        self.prior = Prior(v_dim, d_dim, pad_index)
        self.posterior = Posterior(v_dim, d_dim, h_dim, pad_index)
        self.standard_normal = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

    def forward(self, center, pos_c, neg_c, mask, margin=10.0):
        """The model samples an encoding from the posterior."""
        # Sample mean and sigma from the posterior of central word based on its true context
        mu_posterior, sigma_posterior = self.posterior(center, pos_c, mask)
        mu_prior_cen, sigma_prior_cen = self.prior(center)
        # [B x D], [B x 1]

        # If not used anywhere else, have forward return unsqueezed tensors already
        mu_posterior = mu_posterior.unsqueeze(1)
        mu_prior_cen = mu_prior_cen.unsqueeze(1)
        # [B x 1 x D]

        # Sample mean and sigma from the prior of both positive and negative context for the hinge loss
        mu_prior_pos, sigma_prior_pos = self.prior(pos_c)
        mu_prior_neg, sigma_prior_neg = self.prior(neg_c)
        # [B x W x D], [B x W x 1]

        # Squeeze the sigmas
        sigma_prior_pos = sigma_prior_pos.squeeze()
        sigma_prior_neg = sigma_prior_neg.squeeze()
        # [B x W]

        # Compute KL-divergences
        kl_cen = self._kl_divergence(mu_posterior, sigma_posterior, mu_prior_cen, sigma_prior_cen)
        kl_pos = self._kl_divergence(mu_posterior, sigma_posterior, mu_prior_pos, sigma_prior_pos)
        kl_neg = self._kl_divergence(mu_posterior, sigma_posterior, mu_prior_neg, sigma_prior_neg)

        # Compute hinge-style loss and normalize over batch size. We mask pad context.
        loss = torch.sum(kl_cen + (mask.float() * F.relu(kl_pos - kl_neg + margin)).sum(dim=1)) / mu_posterior.shape[0]

        return loss

    def _kl_divergence(self, mu_1, sigma_1, mu_2, sigma_2):
        """
        Batch wise computation of KL divergence between spherical Gaussians.
        """
        batch_size = mu_1.shape[0]
        embed_size = mu_1.shape[2]
        window_size = mu_2.shape[1]

        # Means inner product with respect to Sigma_2. In: [B x W x D], out: [B x W]
        means_diff = mu_2 - mu_1
        ips2 = torch.matmul(means_diff.view(batch_size, window_size, 1,  embed_size),
                            means_diff.view(batch_size, window_size, embed_size, 1)).view(batch_size, window_size)

        # Compute rest of the KL training instance wise, sum over all instances and immediately average for loss
        # In: [B x W], out: [B x W]
        return 0.5 * (embed_size*(torch.log(sigma_2/sigma_1) - 1 + sigma_1/sigma_2) + ips2/sigma_2)

    def _sample(self, mu, sigma):
        """Reparameterized sampling from a Gaussian density."""
        return mu + sigma * self.standard_normal.sample_n(mu.shape[0])


class Prior(nn.Module):
    """Small feedforward module that outputs the (prior) mean and sigma of a Gaussian."""

    def __init__(self, v_dim, d_dim, pad_index):
        super().__init__()

        # Embeddings for Mu and Sigma, only dependent on the words p(z|w); the Gaussians are spherical
        self.mu = nn.Embedding(v_dim, d_dim, padding_idx=pad_index)
        self.sigma = nn.Embedding(v_dim, 1, padding_idx=pad_index)

    def forward(self, x):
        """Embed word x into a d-dimensional mu and (diagonal) (positive) sigma vector."""
        mu = self.mu(x)
        sigma = F.softplus(self.sigma(x))

        return mu, sigma


class Posterior(nn.Module):
    """Small feedforward module that outputs the (posterior) mean and sigma of a Gaussian."""

    def __init__(self, v_dim, d_dim, h_dim, pad_index):
        super().__init__()

        self.h_dim = h_dim
        self.d_dim = d_dim

        # The first layer is just a random embedding of the center and context words, <pad> gets zero embedding
        self.embedding = nn.Embedding(v_dim, d_dim, padding_idx=pad_index)

        # Mean and sigma of the posterior Gaussians, which are again spherical
        self.linear = nn.Linear(d_dim * 2, h_dim)
        self.mean = nn.Linear(h_dim, d_dim)
        self.sigma = nn.Linear(h_dim, 1)

    def forward(self, center, context, mask):
        """
        The forward pass through the posterior network predicts the mean and sigma of a Normal given a batch of central words
        and their (positive) context words.
        """
        # We expect center to be of shape [B] and context to be of shape [B x W]
        center = self.embedding(center)
        context = self.embedding(context)

        # We need the size of the context window for reshaping
        w_dim = context.shape[1]

        # We repeat the center embeddings W times and flatten context, so we can concat and do a single forward pass
        center = center.unsqueeze(1).repeat(1, w_dim, 1).view(-1, self.d_dim)
        context = context.view(-1, self.d_dim)
        x = torch.cat([center, context], dim=1)

        # Forward pass, masking and sum, result will be a shape [B x H] matrix
        # Mask is of shape [B x W], which we expand to [B x W x H] so we can mask after the forward pass but before sum
        h = F.relu(self.linear(x)).view(-1, w_dim, self.h_dim)
        mask = mask.unsqueeze(2).expand(-1, -1, self.h_dim)
        h_sum = (h * mask.float()).sum(dim=1)

        # Compute mean and sigma
        mean = self.mean(h_sum)
        sigma = F.softplus(self.sigma(h_sum))  # We don't use log_sigma as in the paper for stability reasons

        return mean, sigma
