#!/usr/bin/env python3

import torch

from settings import parse_settings
from helpers import construct_data_path_ea, load_model, construct_model_path


def aer(opt):
    # GPU or CPU selection
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    opt.model = "embedaling"

    # correct v_dim if vocab_size is 0
    if opt.vocab_size == 0:
        opt.v_dim_en = msgpack.load(
            open(osp.join(opt.data_path, opt.dataset, "pad_index_embedalign_{}.en".format(opt.max_sentence_size)), 'rb')) + 1
        opt.v_dim_fr = msgpack.load(
            open(osp.join(opt.data_path, opt.dataset, "pad_index_embedalign_{}.fr".format(opt.max_sentence_size)), 'rb')) + 1

    # Read aer sentences
    with open(osp.join(opt.aer_path, '{}.en'.format(opt.aer_mode)), 'r') as f:
        english = f.read().splitlines()
    with open(osp.join(opt.aer_path, '{}.fr'.format(opt.aer_mode)), 'r') as f:
        french = f.read().splitlines()

    # Open word to index dictionary for embedalign
    word_to_idx = msgpack.load(open(construct_data_path_ea(opt, "wordIndexMap.en"), 'rb'), encoding='utf-8')

    # Add UNKS
    if opt.vocab_size == 0:
        word_to_idx = defaultdict(lambda: opt.v_dim_en - 1, word_to_idx)
    else:
        word_to_idx = defaultdict(lambda: word_to_idx['UNK'], word_to_idx)

    # Process aer sentences
    english_iterator = []
    french_iterator = []
    for e_sen, f_sen in zip(english, french):
        english_iterator.append(torch.tensor([word_to_idx[word]
                                              for word in e_sen.split(' ')], device=device, dtype=torch.long))
        french_iterator.append(torch.tensor([word_to_idx[word]
                                             for word in f_sen.split(' ')], device=device, dtype=torch.long))

    # Construct and load  model
     model = EmbedAlign(opt.v_dim_en, opt.v_dim_fr, opt.d_dim, opt.h_dim,
                               opt.neg_dim, opt.v_dim_en-1, opt.v_dim_fr-1, device).to(device)
     try:
         model = load_model(construct_model_path(opt, True), model)
     except:
         print("No model of type {}, calculating scores with untrained model.".format(opt.model))

     # Forward pass and compute aer
     alignments_file = open(osp.join(opt.out_path, opt.dataset, "{}_alignments".format(opt.aer_mode)), 'w')
     for i, x_en, x_fr in enumerate(zip(english_iterator, french_iterator)):
         alignments = model.get_alignments(x_en, x_fr, x_en.shape[0])

         # Write alignments to file
         for j in range(aligments.shape[1]):
             alignments_file.write('{} {} {}\n'.format(i, alignments[:, j].item(), j))

     alignments_file.close()


if __name__ == '__main__':
    opt = parse_settings()

    aer(opt)
