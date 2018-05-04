#!/usr/bin/python3

import torch
from torch.utils.data import DataLoader

from settings import parse_settings
from dataset import SkipGramData
from Models.skipgram import SkipGram
from Models.bayesian import Bayesian


def main(opt):
    # Load data
    data = DataLoader(SkipGramData(opt.pos_path, opt.neg_path),
                      batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Load model
    if opt.model == "skipgram":
        model = SkipGram(opt.v_dim, opt.h_dim)
    elif opt.model == "bayesian":
        # do something else
        raise NotImplementedError()
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
