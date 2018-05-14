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


class EmbedAlignData(Dataset):

    def __init__(self, data, pad_index):
        with open(data, 'rb') as f:
            data = msgpack.load(f)

        # Extract sentences
        self._en_data = torch.LongTensor([d[0] for d in data])
        self._fr_data = torch.LongTensor([d[1] for d in data])

        # Mask padding
        self._en_mask = 1 - (self._en_data == pad_index).long()
        self._fr_mask = 1 - (self._fr_data == pad_index).long()

        # English sentence length
        self._en_len = self._en_mask.sum(dim=1)

    def __len__(self):
        return self._en_data.shape[0]

    def __getitem__(self, idx):
        return self._en_data[idx], self._en_len[idx], self._en_mask[idx], self._fr_data[idx], self._fr_mask[idx]


def sort_collate(batch):
    """Sort a given batch on its length."""
    # Unpack. Works only with EmbedAlignData
    en_data = batch[0]
    en_len = batch[1]
    en_mask = batch[2]
    fr_data = batch[3]
    fr_mask = batch[4]

    # Get sort indices from the len array
    en_len, indices = torch.sort(en_len, descending=True)
    en_data = en_len[indices]
    en_mask = en_mask[indices]
    fr_data = fr_data[indices]
    fr_mask = fr_mask[indices]

    return (en_data, en_len, en_mask, fr_data, fr_mask)
