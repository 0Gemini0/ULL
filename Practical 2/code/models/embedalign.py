"""
EmbedAlign model in PyTorch.
"""

import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F


class EmbedAlign(nn.Module):
    """
    The EmbedAlign model in PyTorch.
    """

    def __init__(self, v_dim_en, v_dim_fr, h_dim, d_dim, pad_index):
        super().__init__()

        self._encoder = Encoder(v_dim_en, h_dim, d_dim)
        self._decoder = Decoder(v_dim_en, v_dim_fr, d_dim)

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

    def __init__(self, v_dim_en, h_dim, d_dim)
        super().__init__()

        # TODO: MISSING THE CORRECT PADDING WAY FOR THE LSTM TO NOT ENTANGLE WITH PADDING WORDS
        self._gru = torch.nn.GRU(v_dim_en, h_dim, batch_first=True, bidirectional=True)
        self._mu_proj = torch.nn.Linear(h_dim, d_dim)
        self._sig_proj = torch.nn.Linear(h_dim, d_dim)

    def forward(self, x):
        hidden_states, _ = self._gru(x)
        hidden_states = hidden_states[:,:,:hidden_states.shape[2]/2] + hidden_states[:,:,hidden_states.shape[2]/2:]

        mus = self._mu_proj(hidden_states)
        sigmas = F.softplus(self._sig_proj(hidden_states))

        return mus, sigmas



class Decoder(nn.Module):
    """
    The decoder part of EMBEDALIGN.
    """

    def __init__(self, v_dim_en, v_dim_fr, d_dim)
        super().__init__()

        self._en_proj = torch.nn.Linear(d_dim, v_dim_en)
        self._fr_proj = torch.nn.Linear(d_dim, v_dim_fr)

    def forward(zs):
        en_preds = self._en_proj(zs)
        fr_preds = self._fr_proj(zs)

        return en_preds, fr_preds
