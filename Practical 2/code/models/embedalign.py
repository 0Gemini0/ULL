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

    def __init__(self, v_dim, d_dim, h_dim, pad_index):
        super().__init__()

        # TODO: layers

    def forward(self, center, pos_c, mask):
