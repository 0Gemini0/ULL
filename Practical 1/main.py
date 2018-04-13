#!/usr/bin/python3

"""
This file will contain the main script.
"""
import numpy as np
from settings import Settings

from helpers import load_embeddings, get_cosine, get_correlation, retrieve_SIMLEX999_data_dict, retrieve_MEN_data_dict, compute_correlations, reduce_dimensions, visualize_embeddings_2d


def main(opt):

    # Load word embeddings into dictionaries
    deps = load_embeddings("Embeddings/deps.words")
    bow2 = load_embeddings("Embeddings/bow2.words")
    bow5 = load_embeddings("Embeddings/bow5.words")


    if (opt.exercise == 3):

        # Load similarity dataset into dictionaries
        simlex = retrieve_SIMLEX999_data_dict("Similarities/SimLex-999.txt")
        men = retrieve_MEN_data_dict('Similarities/MEN_dataset_natural_form_full')

        # Test with cosine, pearson, spearman, simlex, MEN
        p_sim, s_sim = compute_correlations([deps, bow2, bow5], simlex)
        p_men, s_men = compute_correlations([deps, bow2, bow5], men)

        print("pearson: {}, spearman: {} for SimLex".format(p_sim, s_sim))
        print("pearson: {}, spearman: {} for MEN".format(p_men, s_men))

    elif (opt.exercise == 4):
        pass

    elif (opt.exercise == 5):
        # Reduce dimensions of deps
        pca, tsne = reduce_dimensions(deps, 2, 50, True)

        # Visualize
        visualize_embeddings_2d(pca, "PCA reduced embeddings.", 1000)
        visualize_embeddings_2d(tsne, "TSNE reduced embeddings.", 1000)




if __name__ == '__main__':
    opt = Settings.args
    main(opt)
