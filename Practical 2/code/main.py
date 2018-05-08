#!/usr/bin/python3

import torch
from torch.utils.data import DataLoader

from settings import parse_settings
from dataset import SkipGramData
from models.skipgram import SkipGram
from models.bayesian import Bayesian


def construct_path(opt, name):
    return osp.join(opt.data_path, opt.dataset, "training_" + opt.vocab_size + "_" + str(bool(opt.lowercase))
                    + "_" + opt.window_size + "_" + opt.k + "_" + name + "." + opt.language)


def main(opt):
    # Load data
    data = DataLoader(SkipGramData(construct_path(opt, "samples"), construct_path(opt, "negativeSamples")),
                      batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    idx_to_word = msgpack.load(open(construct_path(opt, "indexWordMap"), 'rb'), encoding='utf-8')

    # Load model
    if opt.model == "skipgram":
        model = SkipGram(opt.v_dim, opt.d_dim)
    elif opt.model == "bayesian":
        model = Bayesian(opt.v_dim, opt.d_dim, opt.h_dim)
    elif opt.model == "embedalign":
        # do something else else
        raise NotImplementedError()
    else:
        raise Exception("Model not recognized, choose [skipgram, bayesian, embedalign]")

    # Define optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(parameters, opt.lr)

    # Training loop
    for i in range(opt.num_epochs):
        for center, pos_context, pos_mask, neg_context, neg_mask in data:
            loss = model(center, pos_context, pos_mask, neg_context, neg_mask)

            # Get gradients and update parameters
            loss.backward()
            optimizer.step()


if __name__ = '__main__':
    opt = parse_settings()
    main(opt)
