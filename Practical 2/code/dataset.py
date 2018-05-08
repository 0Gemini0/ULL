import torch
from torch.utils.data import DataSet
import msgpack


class SkipGramData(DataSet):

    def __init__(self, pos_path, neg_path, pad_index=-1):
        positive_data = msgpack.load(open(pos_path, 'rb'))
        negative_data = msgpack.load(open(neg_path, 'rb'))

        # Extract words and context to tensors
        self._center = torch.LongTensor([data[0] for data in positive_data])
        self._pos_context = torch.LongTensor(data[1][0] + data[1][1] for data in positive_data)
        self._neg_context = torch.LongTensor(data[1] for data in negative_data)

        # Mask padding
        self._pos_mask = 1 - (pos_context == pad_index).long()
        self._neg_mask = 1 - (neg_context == pad_index).long()

    def __len__(self):
        return self._center.shape[0]

    def __getitem__(self, idx):
        return self._center[idx], self._pos_context[idx], self._pos_mask[idx], self._neg_context[idx], self._neg_mask[idx]
