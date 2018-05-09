import torch
from torch.utils.data import Dataset
import msgpack


class SkipGramData(Dataset):

    def __init__(self, pos_path, neg_path, pad_index):
        with open(pos_path, 'rb') as f:
            positive_data = msgpack.load(f)
        with open(neg_path, 'rb') as f:
            negative_data = msgpack.load(f)

        # Extract words and context to tensors
        self._center = torch.LongTensor([data[0] for data in positive_data])
        self._pos_context = torch.LongTensor([data[1][0] + data[1][1] for data in positive_data])
        self._neg_context = torch.LongTensor([data[1] for data in negative_data])

        # Mask padding
        self._pos_mask = 1 - (self._pos_context == pad_index).long()
        self._neg_mask = 1 - (self._neg_context == pad_index).long()

        print(sum(sum(self._pos_mask - self._neg_mask)))

    def __len__(self):
        return self._center.shape[0]

    def __getitem__(self, idx):
        return self._center[idx], self._pos_context[idx], self._pos_mask[idx], self._neg_context[idx], self._neg_mask[idx]
