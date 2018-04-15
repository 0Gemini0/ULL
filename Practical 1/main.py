#!/usr/bin/python3

"""
This file will contain the main script.
"""
import numpy as np
from settings import Settings

from helpers import load_embeddings
from helpers import retrieve_SIMLEX999_data_dict, retrieve_MEN_data_dict, compute_correlations
from helpers import reduce_dimensions, visualize_embeddings, cluster, save_clusters

def main(opt):

    # # Load word embeddings into dictionaries
    deps = load_embeddings("Embeddings/deps.words")
    bow2 = load_embeddings("Embeddings/bow2.words")
    bow5 = load_embeddings("Embeddings/bow5.words")

    if (opt.exercise == 3 | opt.exercise == 1):
        # Load similarity dataset into dictionaries
        simlex = retrieve_SIMLEX999_data_dict("Similarities/SimLex-999.txt")
        men = retrieve_MEN_data_dict('Similarities/MEN_dataset_natural_form_full')

        # Test with cosine, pearson, spearman, simlex, MEN
        p_sim, s_sim = compute_correlations([deps, bow2, bow5], simlex)
        p_men, s_men = compute_correlations([deps, bow2, bow5], men)

        print("pearson: {}, spearman: {} for SimLex".format(p_sim, s_sim))
        print("pearson: {}, spearman: {} for MEN".format(p_men, s_men))

    elif (opt.exercise == 4 | opt.exercise == 1):
        pass

    elif (opt.exercise == 5 or opt.exercise == 1):
        # Load nouns
        with open(opt.nouns, 'r', encoding='utf8') as f:
            nouns = f.read().split()

        # Return embedding matrices ordered as the nouns list
        deps_nouns = np.array([deps[noun] for noun in nouns])
        bow2_nouns = np.array([bow2[noun] for noun in nouns])
        bow5_nouns = np.array([bow5[noun] for noun in nouns])

        # Reduce dimensions of noun embeddings
        reduced_embeddings = reduce_dimensions([deps_nouns, bow2_nouns, bow5_nouns],
                                      opt.dim, opt.red_mode, opt.verbose, opt.tsne_dim, opt.tsne_num)

        # Clustering of noun embeddings
        labels = cluster([deps_nouns, bow2_nouns, bow5_nouns],
                         opt.clu_mode, opt.verbose, opt.k, opt.eps, opt.min_samples)

        # Visualize
        titles = ["{} deps embeddings with {} based cluster labels".format(opt.red_mode.capitalize(), opt.clu_mode),
                  "{} bow2 embeddings with {} based cluster labels".format(opt.red_mode.capitalize(), opt.clu_mode),
                  "{} bow5 embeddings with {} based cluster labels".format(opt.red_mode.capitalize(), opt.clu_mode)]
        visualize_embeddings(reduced_embeddings, labels, titles,
                            opt.viz_num, opt.dim, opt.verbose)

        # Qualitative
        save_clusters(nouns, labels, opt.out, ['deps_clusters.txt', 'bow2_cluster.txt', 'bow5_cluster.txt'])


if __name__ == '__main__':
    opt = Settings.args
    main(opt)
