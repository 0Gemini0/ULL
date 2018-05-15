"""
EmbedAlign model in PyTorch.
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Normal, Categorical
import torch.nn.functional as F


class EmbedAlign(nn.Module):
    """
    The EmbedAlign model in PyTorch.
    """

    def __init__(self, v_dim_en, v_dim_fr, d_dim, h_dim, neg_dim, pad_index, device):
        super().__init__()

        self._encoder = Encoder(v_dim_en, h_dim, d_dim, pad_index)
        self._decoder = Decoder(v_dim_en, v_dim_fr, d_dim, neg_dim, pad_index, device)
        self.sparse_params = self._encoder.sparse_params + self._decoder.sparse_params
        self.device = device

    def forward(self, x_en, x_fr, en_mask, fr_mask, en_len):
        # Encode the english sentence into Gaussian parameters
        mus, sigmas = self._encoder(x_en, en_len)

        # Sample zs from the Gaussians with reparametrization
        zs = self._sample(mus, sigmas)

        # Decode the samples into estimated word and alignment probabilities with CSS
        en_probs, fr_probs = self._decoder(zs, x_en, x_fr)

        # Sum french probs, take log, mask, and sum some more, yielding the reconstruction loss per sentence. Alos get the english log probs, masked
        en_log_probs = torch.log(en_probs) * en_mask
        fr_rec_loss = (torch.log(fr_probs.sum(dim=2)) * fr_mask).sum(dim=1) / en_len

        # Compute KL part of the loss, masked for padding
        kl = self._kl_divergence(mus, sigmas) * en_mask

        # Return final ELBO, averaged over the minibatch
        return (fr_rec_loss + (en_log_probs - kl * en_mask).sum(dim=1)).sum() / x_en.shape[0]

    def get_alignments(self, x_en, x_fr, en_mask, fr_mask):
        """Using parts of the forward pass we can extract predicted alignments from the model."""
        # Encode the english sentence to Gaussian parameters
        mus, sigmas = self._encoder(x_en)

        # Sample zs from the Gaussians with reparametrization
        zs = self._sample(mus, sigmas)

        # Decode the samples into estimated word and alignment probabilities with CSS
        _, fr_probs = self._decoder(zs, x_en, x_fr, en_mask, fr_mask)

        # Transform fr_probs into alignments
        _, alignments = torch.max(fr_probs, dim=2)

        return alignments

    def _sample(self, mu, sigma):
        """Reparameterized sampling from a Gaussian density."""
        return mu + sigma * self.standard_normal.sample_n(mu.shape)

    def _kl_divergence(self, mu, sigma):
        """
        Batch wise computation of KL divergence between diagonal Gaussian and Unit Gaussian.
        """
        return -0.5 * (2 * torch.log(sigma) - sigma ** 2 - mu ** 2 + 1).sum(dim=2)


class Encoder(nn.Module):
    """
    The encoder part of EmbedAlign.
    """

    def __init__(self, v_dim_en, h_dim, d_dim, pad_index):
        super().__init__()

        self._det_emb = nn.Embedding(v_dim_en, d_dim, padding_idx=pad_index, sparse=True)
        self.sparse_params = [p for p in self.parameters()]

        self._gru = nn.GRU(v_dim_en, h_dim, batch_first=True, bidirectional=True)
        self._mu_proj = torch.nn.Linear(h_dim, d_dim)
        self._sig_proj = torch.nn.Linear(h_dim, d_dim)

    def forward(self, x, x_len):
        x_emb = self._det_emb(x)
        x_packed = pack_padded_sequence(x, x_len, batch_first=True)
        # [B * S x D]

        h_packed, _ = self._gru(x)
        h = pad_packed_sequence(h_packed, batch_first=True)
        # [B x S x 2H]

        h_sum = h[:, :, :h.shape[2]/2] + h[:, :, h.shape[2]/2:]
        # [B x S x H]

        mus = self._mu_proj(h_sum)
        sigmas = F.softplus(self._sig_proj(h_sum))
        # [B x S x D]

        return mus, sigmas


class Decoder(nn.Module):
    """
    The decoder part of EmbedAlign.
    """

    def __init__(self, v_dim_en, v_dim_fr, d_dim, neg_dim, pad_index, device):
        super().__init__()
        self.v_dim_en = v_dim_en
        self.v_dim_fr = v_dim_fr
        self.neg_dim = neg_dim
        self.device = device

        self.en_embedding = nn.Embedding(v_dim_en, d_dim, padding_idx=pad_index, sparse=True)
        self.fr_embedding = nn.Embedding(v_dim_fr, d_dim, padding_idx=pad_index, sparse=True)
        self.sparse_params = [p for p in self.parameters()]

    def forward(self, zs, x_en, x_fr):
        # Use complementary sum sampling to approximate the probabilities of the french and english sentences given z
        en_probs = self._css(x_en, self.v_dim_en, self.neg_dim, self.en_embedding, zs, "en")
        fr_probs = self._css(x_fr, self.v_dim_fr, self.neg_dim, self.fr_embedding, zs, "fr")

        return en_probs, fr_probs

    def _css(self, x, v_dim, num, embedding, z, language):
        """Generate a negative set without replacement for CSS given a batch as positive set."""
        positive_set, _ = np.unique(x.numpy())
        neg_dim = v_dim - positive_set.shape[0]
        weights = torch.ones([v_dim], device=self.device)
        weights[positive_set] = 0.

        negative_set = torch.multinomial(weights, num, replacement=False)
        kappa = torch.tensor(neg_dim / num, device=self.device)

        batch_embeddings = embedding(x)
        positive_embeddings = embedding(torch.tensor(positive_embeddings, device=self.device))
        negative_embeddings = embedding(negative_set)

        # TODO: stable exponentials
        if language == "en":
            batch_score = torch.exp((z * batch_embeddings).sum(dim=2))
        else:
            batch_score = torch.exp(torch.bmm(batch_embeddings, z.transpose(1, 2)))
        positive_score = torch.exp(torch.matmul(z, positive_embeddings.transpose(1, 0))).sum(dim=2)
        negative_score = kappa * torch.exp(torch.matmul(z, negative_embeddings.transpose(1, 0))).sum(dim=2)

        if language == "en":
            return batch_score / (positive_score + negative_score)
        else:
            return batch_score / (positive_score + negative_score).unsqueeze(1)
