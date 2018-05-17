#!/usr/bin/env python3

import os.path as osp
from collections import defaultdict

import numpy as np
import torch
import msgpack
import torch.nn.functional as f

from settings import parse_settings
from models.skipgram import SkipGram
from models.bayesian import Bayesian
from models.embedalign import EmbedAlign
from helpers import construct_data_path, construct_data_path_ea, construct_model_path, load_model


def cosine_distance(target_candidates):
    # Normalise embeddings.
    target_candidates = f.normalize(target_candidates, p=1, dim=1)

    # Get target and substitution candidates.
    target = target_candidates[0]
    candidates = target_candidates[1:]

    # Compute inner product. Due to the vectors being normalised it is the same as cosine distance.
    inner_prod = candidates.mm(target.unsqueeze(1))

    # Sort results on similarity and get the correct indices
    sorted_values = torch.sort(inner_prod.squeeze(), descending=True)
    sorted_values_indices = (sorted_values[0], sorted_values[1] + 1)

    return sorted_values_indices


def kl_distance(tuple_means_sigmas):
    # Get target and substitution candidates.
    target_mean = tuple_means_sigmas[0][0]
    candidates_means = tuple_means_sigmas[0][1:]
    target_sigma = tuple_means_sigmas[1][0]
    candidates_sigmas = tuple_means_sigmas[1][1:]

    # Compute the logarithms of the determinants
    logs = torch.log(torch.prod(candidates_sigmas, dim=1)/torch.prod(target_sigma))

    # Compute the trace of candidates' cov inverses dot target cov
    traces = torch.sum(torch.mm(1.0/candidates_sigmas, target_sigma.unsqueeze(1)), dim=1)

    # Compute inner products wrt sigmas
    inner_products_wrt_sigmas = torch.sum((target_mean-candidates_means) *
                                          candidates_sigmas*(target_mean-candidates_means), dim=1)

    # Compute final KLs
    KLs = 0.5*(logs - target_mean.shape[0] + traces + inner_products_wrt_sigmas)

    # Sort results on KL divergences and get the correct indices
    sorted_values = torch.sort(KLs.squeeze(), descending=True)
    sorted_values_indices = (sorted_values[0], sorted_values[1] + 1)

    return sorted_values_indices


def write_lst(filename, data, scores, indices, model, idx_to_word):
    with open(filename, 'a') as f:
        f.write("RANKED\t{} {}".format(data[3], data[4]))

        for score, index in zip(scores, indices):
            if model == 'embedalign':
                candidate = idx_to_word[int(data[0][index, data[2].item()])]
            else:
                candidate = idx_to_word[int(data[0][index].item())]
            f.write("\t{} {}".format(candidate, score))

        f.write("\n")


def lst_preprocess(opt, device, model):
    # Read lst data
    with open(osp.join(opt.lst_path, 'lst_test.preprocessed'), 'r') as f:
        lst_data = [sentence.split('\t') for sentence in f.read().splitlines()]

    # Open word to index dictionary
    if model != "embedalign":
        word_to_idx = msgpack.load(open(construct_data_path(opt, "wordIndexMap"), 'rb'), encoding='utf-8')
    else:
        word_to_idx = msgpack.load(open(construct_data_path_ea(opt, "wordIndexMap.en"), 'rb'), encoding='utf-8')

    with open(osp.join(opt.lst_path, 'lst.gold.candidates'), 'r') as f:
        target_candidates = f.read().splitlines()

    # Read candidates into a dictionary, removing mw expressions and unknown candidates
    candidate_dict = {}
    for target_candidate in target_candidates:
        target, candidates = target_candidate.split("::")
        candidates = [candidate for candidate in candidates.split(";") if candidate in word_to_idx]
        candidate_dict[target] = candidates

    # Change the word_to_idx dict to a defaultdict so we can catch unks
    if opt.vocab_size == 0:
        word_to_idx = defaultdict(lambda: opt.v_dim_en - 1, word_to_idx)
    else:
        word_to_idx = defaultdict(lambda: word_to_idx['UNK'], word_to_idx)

    # Construct target and candidate sentences
    sentence_iterator = []
    len_iterator = []
    position_iterator = []
    context_iterator = []
    context_mask_iterator = []
    center_iterator = []
    target_iterator = []
    id_iterator = []
    for data in lst_data:
        # Unpack the lst data
        target = data[0]
        id = data[1]
        position = int(data[2])
        target_sentence = [word_to_idx[word] for word in data[3].split(" ")]

        # Add some usefull data to the iterators
        position_iterator.append(torch.tensor(position, device=device, dtype=torch.long))
        target_iterator.append(target)
        id_iterator.append(id)

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
        pad_index = opt.v_dim_en - 1
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

    # Return data for the correct model
    if model == "embedalign":
        return sentence_iterator, len_iterator, position_iterator, target_iterator, id_iterator
    else:
        return center_iterator, context_iterator, context_mask_iterator, target_iterator, id_iterator


def lst(opt):
    # GPU or CPU selection
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ########################## SKIPGRAM AND BAYESIAN ################################
    for model_name in ["skipgram", "bayesian", "embedalign"]:
        opt.model = model_name

        # When the vocab size is 0, the entire vocab is used and its size loaded from disk.
        if opt.vocab_size == 0:
            if opt.model == "embedalign":
                opt.v_dim_en = msgpack.load(
                    open(osp.join(opt.data_path, opt.dataset, "pad_index_embedalign_{}.en".format(opt.max_sentence_size)), 'rb')) + 1
                opt.v_dim_fr = msgpack.load(
                    open(osp.join(opt.data_path, opt.dataset, "pad_index_embedalign_{}.fr".format(opt.max_sentence_size)), 'rb')) + 1
            else:
                opt.v_dim_en = msgpack.load(
                    open(osp.join(opt.data_path, opt.dataset, "pad_index_skipgram.en"), 'rb')) + 1
                opt.v_dim_fr = msgpack.load(
                    open(osp.join(opt.data_path, opt.dataset, "pad_index_skipgram.en"), 'rb')) + 1

        # Preprocess the lst files to get torch format data
        data_in = lst_preprocess(opt, device, opt.model)

        # Open index to word list
        if opt.model != "embedalign":
            idx_to_word = msgpack.load(open(construct_data_path(opt, "indexWordMap"), 'rb'), encoding='utf-8')
        else:
            idx_to_word = msgpack.load(open(construct_data_path_ea(opt, "indexWordMap.en"), 'rb'), encoding='utf-8')

        if opt.model == "skipgram":
            model = SkipGram(opt.v_dim_en, opt.d_dim, opt.v_dim_en-1).to(device)
        elif opt.model == "bayesian":
            model = Bayesian(opt.v_dim_en, opt.d_dim, opt.h_dim, opt.v_dim_en-1).to(device)
        elif opt.model == "embedalign":
            model = EmbedAlign(opt.v_dim_en, opt.v_dim_fr, opt.d_dim, opt.h_dim,
                               opt.neg_dim, opt.v_dim_en-1, opt.v_dim_fr-1, opt.kl_step, device).to(device)

        # We give a warning when no model can be loaded
        # try:
        model = load_model(construct_model_path(opt, True), model)
        # except:
        #     print("No model of type {}, calculating scores with untrained model.".format(opt.model))

        # Storage paths
        cosine_file = osp.join(opt.out_path, opt.dataset, "{}_cos_lst.predictions".format(opt.model))
        kl_file = osp.join(opt.out_path, opt.dataset, "{}_kl_lst.predictions".format(opt.model))

        # Get the scores from the model and write to file
        for data in zip(data_in[0], data_in[1], data_in[2], data_in[3], data_in[4]):
            # Special forward pass to get mus and sigmas (embeddings)
            embeddings = model.lst_pass(data)

            # We score the embeddings using both cosine distance and KL whenever possible
            cos_scores, cos_indices = cosine_distance(embeddings[0])
            write_lst(cosine_file, data, cos_scores, cos_indices, opt.model, idx_to_word)
            if opt.model != "skipgram":
                kl_scores, kl_indices = kl_distance(embeddings)
                write_lst(kl_file, data, kl_scores, kl_indices, opt.model, idx_to_word)


if __name__ == "__main__":
    opt = parse_settings()

    lst(opt)
