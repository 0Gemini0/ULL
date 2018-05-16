#!/usr/bin/env python3
import os.path as osp
from time import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SparseAdam
from torch.autograd import Variable
import msgpack
import numpy as np

from settings import parse_settings
from dataset import SkipGramData, EmbedAlignData, sort_collate
from models.skipgram import SkipGram
from models.bayesian import Bayesian
from models.embedalign import EmbedAlign
from helpers import construct_data_path, construct_data_path_ea, construct_model_path, save_checkpoint, load_checkpoint


def main(opt):
    # We activate the GPU if cuda is available, otherwise computation will be done on cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # When the vocab size is 0, the entire vocab is used and its size loaded from disk.
    if opt.vocab_size == 0:
        if opt.model == "embedalign":
            opt.v_dim_en = msgpack.load(
                open(osp.join(opt.data_path, opt.dataset, "pad_index_embedalign_{}.en".format(opt.max_sentence_size)), 'rb')) + 1
            opt.v_dim_fr = msgpack.load(
                open(osp.join(opt.data_path, opt.dataset, "pad_index_embedalign_{}.fr".format(opt.max_sentence_size)), 'rb')) + 1
        else:
            opt.v_dim_en = msgpack.load(open(osp.join(opt.data_path, opt.dataset, "pad_index_skipgram.en"), 'rb')) + 1
            opt.v_dim_fr = msgpack.load(open(osp.join(opt.data_path, opt.dataset, "pad_index_skipgram.en"), 'rb')) + 1

    # Now we load the data fitting the selected model
    print("Loading Data...")
    if opt.model == "embedalign":
        data = DataLoader(EmbedAlignData(construct_data_path_ea(opt, "data.both"), opt.v_dim_en - 1, opt.v_dim_fr - 1),
                          batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=sort_collate)
    else:
        data = DataLoader(SkipGramData(construct_data_path(opt, "samples"), construct_data_path(opt, "negativeSamples"),
                                       opt.v_dim_en - 1), batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print("Data was succesfully loaded.")

    # We load the selected model and place it on the available device(s)
    if opt.model == "skipgram":
        model = SkipGram(opt.v_dim_en, opt.d_dim, opt.v_dim_en-1)
    elif opt.model == "bayesian":
        model = Bayesian(opt.v_dim_en, opt.d_dim, opt.h_dim, opt.v_dim_en-1)
    elif opt.model == "embedalign":
        model = EmbedAlign(opt.v_dim_en, opt.v_dim_fr, opt.d_dim, opt.h_dim,
                           opt.neg_dim, opt.v_dim_en-1, opt.v_dim_fr-1, device)
    else:
        raise Exception("Model not recognized, choose [skipgram, bayesian, embedalign]")

    if opt.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        sparse_params = model.module.sparse_params
        print("Using parallel processing on GPUs")
    else:
        model.to(device)
        sparse_params = model.sparse_params

    # To work with ADAM and Sparse Embeddings, we need (retardedly) to manually store sparse parameters

    parameters_sparse = list(filter(lambda p: p.requires_grad, sparse_params))
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    parameters = [p for p in parameters if not any(p is p_ for p_ in parameters_sparse)]
    optimizers = []
    if parameters_sparse:
        optimizers.append(SparseAdam(parameters_sparse, opt.lr))
    if parameters:
        optimizers.append(Adam(parameters, opt.lr))

    losses = []
    best_loss = np.inf
    for i in range(opt.num_epochs):
        ep_loss = 0.
        t = 0.
        is_best = False
        for j, data_in in enumerate(data):
            start = time()
            # No longer tedious! Send data to selected device
            data_in = [inp.to(device) for inp in data_in]

            # Actual training
            loss = model(data_in)

            ep_loss += loss.item()

            # Get gradients and update parameters
            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            # See progress
            if j % 10 == 0:
                print("\rSteps this epoch: {}, time: {}s, avg_loss: {}".format(
                    j, t, ep_loss/(j+1)), end="", flush=True)

            t += time() - start

        avg_loss = ep_loss/(j+1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            is_best = True
        losses.append(avg_loss)

        # Save_checkpoint
        save_checkpoint(opt, model, optimizers, i, losses, is_best)
        print("Epoch: {}, Average Loss: {}".format(i, avg_loss))


if __name__ == '__main__':
    opt = parse_settings()
    main(opt)
