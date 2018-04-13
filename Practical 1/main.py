#!/usr/bin/python3

"""
This file will contain the main script.
"""
import numpy as np
from settings import Settings

from helpers import load_embeddings, get_cosine, get_correlation, retrieve_SIMLEX999_data_dict, retrieve_MEN_data_dict, compute_correlations, reduce_dimensions, visualize_embeddings_2d, visualize_embeddings_3d, cluster


def main(opt):

    # Load word embeddings into dictionaries
    deps = load_embeddings("Embeddings/deps.words")
    # bow2 = load_embeddings("Embeddings/bow2.words")
    # bow5 = load_embeddings("Embeddings/bow5.words")


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
        # Load nouns
        with open(opt.nouns, 'r', encoding='utf8') as f:
            nouns = f.read().split()

        # Return an embedding matrix ordered as the nouns list
        deps_nouns = np.array([deps[noun] for noun in nouns])
        print(deps_nouns.shape)
        print(deps_nouns[1,:10], deps_nouns[2,:10])

        # Reduce dimensions of deps
        pca, tsne = reduce_dimensions(deps_nouns, opt.dim, 50, opt.tsne_num, True)

        # K-means
        pca_labels = cluster(pca, k = opt.k)
        tsne_labels = cluster(tsne, k=opt.k)

        # Visualize
        if opt.dim == 2:
            visualize_embeddings_2d(pca, pca_labels, "PCA reduced embeddings.", opt.tsne_num)
            visualize_embeddings_2d(tsne, tsne_labels, "TSNE reduced embeddings.", opt.tsne_num)
        elif opt.dim == 3:
            visualize_embeddings_3d(pca, pca_labels, "PCA reduced embeddings.", opt.tsne_num)
            visualize_embeddings_3d(tsne, tsne_labels, "TSNE reduced embeddings.", opt.tsne_num)
        else:
            print("Cannot visualize in {} dimensions.".format(opt.dim))

        # Qualitative
        for i in range(opt.k):
            print("Cluster {}: {}".format(i, [nouns[j] for j in list(np.argwhere(tsne_labels == i).flatten())]))





if __name__ == '__main__':
    opt = Settings.args
    main(opt)
