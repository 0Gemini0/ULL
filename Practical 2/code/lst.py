#!/usr/bin/env python3

import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import msgpack

from settings import parse_settings
# from models.skipgram import SkipGram
# from models.bayesian import Bayesian
# from models.embedalign import EmbedAlign


def construct_data_path(opt, name):
    return osp.join(opt.data_path, opt.dataset, "training_" + str(opt.vocab_size) + "_" + str(bool(opt.lowercase))
                    + "_" + str(opt.window_size) + "_" + str(opt.k) + "_" + name + "." + opt.language)


def lst(opt):
    # GPU or CPU selection
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Read lst data
    with open(osp.join(opt.lst_path, 'lst_test.preprocessed'), 'r') as f:
        lst_data = [sentence.split('\t') for sentence in f.read().splitlines()]

    # Open word to index dictionary
    word_to_idx = msgpack.load(open(construct_data_path(opt, "wordIndexMap"), 'rb'), encoding='utf-8')
    idx_to_word = msgpack.load(open(construct_data_path(opt, "indexWordMap"), 'rb'), encoding='utf-8')

    with open(osp.join(opt.lst_path, 'lst.gold.candidates'), 'r') as f:
        target_candidates = f.read().splitlines()

    # Read candidates into a dictionary, removing mw expressions and unknown candidates
    candidate_dict = {}
    for target_candidate in target_candidates:
        target, candidates = target_candidate.split("::")
        candidates = [candidate for candidate in candidates.split(";") if candidate in word_to_idx]
        candidate_dict[target] = candidates

    # Change the word_to_idx dict to a defaultdict so we can catch unks
    word_to_idx = defaultdict(lambda: word_to_idx['UNK'], word_to_idx)

    # Construct target and candidate sentences
    sentence_iterator = []
    len_iterator = []
    context_iterator = []
    context_mask_iterator = []
    center_iterator = []
    for data in lst_data:
        # Unpack the lst data
        target = data[0]
        position = int(data[2])
        target_sentence = [word_to_idx[word] for word in data[3].split(" ")]

        # Construct candidate sentences with each target sentence
        target_prev = target_sentence[:position]
        target_post = target_sentence[position+1:]
        candidate_sentences = [target_prev + [word_to_idx[candidate]] +
                               target_post for candidate in candidate_dict[target]]
        sentences = [target_sentence] + candidate_sentences

        # Construct sentence, len and central tensor
        sentence_iterator.append(torch.tensor(sentences, device=device, dtype=torch.long))
        len_iterator.append(torch.tensor([len(target_sentence)] * len(sentences), device=device, dtype=torch.long))
        center_iterator.append(torch.tensor(np.array(sentences)[:, position], device=device, dtype=torch.long))

        # Construct context tensor with padding, and mask
        pad_index = opt.v_dim - 1
        prev_context = np.array(sentences)[:, position-opt.window_size:position]
        post_context = np.array(sentences)[:, position+1:position+1+opt.window_size]
        if prev_context.shape[1] < opt.window_size:
            prev_context = np.concatenate([np.ones([prev_context.shape[0], opt.window_size -
                                                    prev_context.shape[1]], dtype=np.int64) * pad_index, prev_context], axis=1)
        if post_context.shape[1] < opt.window_size:
            post_context = np.concatenate([post_context, np.ones([post_context.shape[0], opt.window_size -
                                                                  post_context.shape[1]], dtype=np.int64) * pad_index], axis=1)
        context = np.concatenate([prev_context, post_context], axis=1)
        context_iterator.append(torch.tensor(context, device=device, dtype=torch.long))
        context_mask_iterator.append(torch.tensor(1 - (context == pad_index), device=device, dtype=torch.long))

    print(len(sentence_iterator), sentence_iterator[0].shape)
    print(sentence_iterator[0].shape,
          len_iterator[0].shape,
          context_iterator[0].shape,
          context_mask_iterator[0].shape,
          center_iterator[0].shape)


if __name__ == "__main__":
    opt = parse_settings()

    lst(opt)
