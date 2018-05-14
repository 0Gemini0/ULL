"""
EmbedAlign model in PyTorch.
"""

import torch
from torch import nn
from torch.nn.utils import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Normal
import torch.nn.functional as F


class EmbedAlign(nn.Module):
    """
    The EmbedAlign model in PyTorch.
    """

    def __init__(self, v_dim_en, v_dim_fr, d_dim, h_dim, pad_index):
        super().__init__()

        self._encoder = Encoder(v_dim_en, h_dim, d_dim, pad_index)
        self._decoder = Decoder(v_dim_en, v_dim_fr, d_dim)
        self.sparse_params = self._encoder.sparse_params

    def forward(self, center, pos_c, mask):
        mus, sigmas = self._encoder(x)
        zs = self._sample(mus, sigmas)
        en_preds, fr_preds = self._decoder(zs)

    def _sample(self, mu, sigma):
        """Reparameterized sampling from a Gaussian density."""
        return mu + sigma * self.standard_normal.sample_n(mu.shape)


class Encoder(nn.Module):
    """
    The encoder part of EMBEDALIGN.
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

        # TODO: Mask here?

        return mus, sigmas


class Decoder(nn.Module):
    """
    The decoder part of EMBEDALIGN.
    """

    def __init__(self, v_dim_en, v_dim_fr, d_dim):
        super().__init__()

        self._en_proj = torch.nn.Linear(d_dim, v_dim_en)
        self._fr_proj = torch.nn.Linear(d_dim, v_dim_fr)

    def forward(self, zs):
        en_preds = self._en_proj(zs)
        fr_preds = self._fr_proj(zs)

        return en_preds, fr_preds
