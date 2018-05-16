#!/usr/bin/env python3

import torch

from settings import parse_settings
from helpers import construct_data_path_ea, load_model, construct_model_path


def aer(opt):
    # Read aer sentences
    with open(osp.join(opt.aer_path, '{}.en'.format(opt.aer_mode)), 'r') as f:
        english = f.read().splitlines()
    with open(osp.join(opt.aer_path, '{}.fr'.format(opt.aer_mode)), 'r') as f:
        french = f.read().splitlines()

    # Open word to index dictionary for embedalign
    word_to_idx = msgpack.load(open(construct_data_path_ea(opt, "wordIndexMap.en"), 'rb'), encoding='utf-8')

    # Process aer sentences
    for e_sen, f_sen in zip(english, french):


if __name__ == '__main__':
    opt = parse_settings()

    aer(opt)
