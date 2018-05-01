"""
Skipgram model in PyTorch.
"""
import torch
from torch import nn
import torch.functional as F


class SkipGram(nn.Module):
    """SkipGram model in PyTorch."""

    def __init__(self, v_dim, h_dim):
        # Embedding matrices for both context and center words
        self.center_embedding = nn.Embedding(v_dim, h_dim)
        self.context_embedding = nn.Embeding(v_dim, h_dim)

    def forward(self, center, pos_c, neg_c):
        # Embed; center and pos_c should be [B x V], neg_c should be [B x K x V]
        center = self.center_embedding(center)
        pos_c = self.context_embedding(pos_c)
        neg_c = self.context_embedding(neg_c)

        # For efficient matrix multiplication with bmm, we add extra singleton dimensions to the batched embeddings
        center = center.unsqueeze(2)
        pos_c = pos_c.unsqueeze(1)

        # Compute scores; score dimension [B]
        # bmm: batch matrix multiply two tensors such that bmm([B x K x V], [B x V x 1]) -> [B, K, 1], effectively
        # computing a row-wise dot product between the rows in the [B x V] embedding matrices.
        pos_scores = F.logsigmoid(torch.bmm(pos_c, center).squeeze())
        neg_scores = F.logsigmoid(torch.bmm(-neg_c, center).squeeze()).sum(dim=1)

        # Directly return the loss, i.e. the negative normalized sum of the positive and negative 'score'
        return -torch.sum(pos_scores + neg_scores) / pos_scores.shape[0]
