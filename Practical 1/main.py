#!/usr/bin/python3

"""
This file will contain the main script.
"""
import numpy as np
from settings import Settings
import os.path as osp

from helpers import load_embeddings
from helpers import retrieve_SIMLEX999_data_dict, retrieve_MEN_data_dict, compute_correlations
from helpers import retrieve_word_analogy_data_dict, normalised_word_embeddings_data
from helpers import create_bStars_preds_data, compute_accuracy_and_MRR
from helpers import reduce_dimensions, visualize_embeddings, cluster, save_clusters


def main(opt):

    # Load word embeddings into dictionaries
    deps = load_embeddings(osp.join(opt.emb_path, "deps.words"))
    bow2 = load_embeddings(osp.join(opt.emb_path, "bow2.words"))
    bow5 = load_embeddings(osp.join(opt.emb_path, "bow5.words"))

    # Easy iteration and printing
    emb_names = ['deps', 'bow2', 'bow5']
    embeddings = [deps, bow2, bow5]

    if (opt.exercise == 3 or opt.exercise == 1):
        # Load similarity dataset into dictionaries
        simlex = retrieve_SIMLEX999_data_dict(osp.join(opt.data_path, "SimLex-999.txt"))
        men = retrieve_MEN_data_dict(osp.join(opt.data_path, 'MEN_dataset_natural_form_full'))

        # Test with cosine, pearson, spearman, simlex, MEN
        p_sim, s_sim, top_sim = compute_correlations(embeddings, simlex, opt.N)
        p_men, s_men, top_men = compute_correlations(embeddings, men, opt.N)

        for i, name in enumerate(emb_names):
            print("pearson: {}, spearman: {} for SimLex with {} embeddings".format(p_sim[i], s_sim[i], name))
            print("pearson: {}, spearman: {} for MEN with {} embeddings".format(p_men[i], s_men[i], name))

        for i, name in enumerate(emb_names):
            print("Top {} most similar pairs on SimLex with {} embeddings.\n {}".format(opt.N, name, top_sim[i]))
            print("Top {} most similar pairs on MEN with {} embeddings.\n {}".format(opt.N, name, top_men[i]))

    elif (opt.exercise == 4 or opt.exercise == 1):
        word_analogy_data = retrieve_word_analogy_data_dict(
            osp.join(opt.data_path, "word-analogy.txt"), lowercase=False)

        for i, dataset in enumerate(embeddings):
            print("\nCurrently working on embedding: " + emb_names[i] + ".")

            normalised_dataset_data = normalised_word_embeddings_data(dataset)
            bStars_preds_data = create_bStars_preds_data(normalised_dataset_data[0], word_analogy_data)

            inner_products = np.dot(normalised_dataset_data[2], bStars_preds_data[1])

            acc_f, mrr_f = compute_accuracy_and_MRR(
                bStars_preds_data[0], normalised_dataset_data[1], inner_products, False)
            acc_t, mrr_t = compute_accuracy_and_MRR(
                bStars_preds_data[0], normalised_dataset_data[1], inner_products, True)

            print("\nEmbedding: " + emb_names[i] + " ||| " +
                  "Accuracy = " + "{:3.2f}".format(acc_t) + "% (" + "{:3.2f}".format(acc_f) + "%) ||| " +
                  "MRR = " + "{:.2f}".format(mrr_t) + " (" + "{:.2f}".format(mrr_f) + ")" +
                  " ||| Total number of queries: " + str(len(bStars_preds_data[0])) + "\n\n\n")

    elif (opt.exercise == 5 or opt.exercise == 1):
        # Load nouns
        with open(osp.join(opt.data_path, 'nouns.txt'), 'r', encoding='utf8') as f:
            nouns = f.read().split()

        # Return embedding matrices ordered as the nouns list
        embeddings_nouns = []
        for embedding in embeddings:
            embeddings_nouns.append(np.array([embedding[noun] for noun in nouns]))

        # Reduce dimensions of noun embeddings
        reduced_embeddings = reduce_dimensions(embeddings_nouns,
                                               opt.dim, opt.red_mode, opt.verbose, opt.tsne_dim, opt.tsne_num)

        # Clustering of noun embeddings
        labels = cluster(embeddings_nouns,
                         opt.clu_mode, opt.verbose, opt.k, opt.eps, opt.min_samples)

        # Visualize
        titles = ["{} {} embeddings with {} based cluster labels".format(
            opt.red_mode.capitalize(), name, opt.clu_mode) for name in emb_names]
        visualize_embeddings(reduced_embeddings, labels, titles,
                             opt.viz_num, opt.dim, opt.verbose)

        # Qualitative
        save_clusters(nouns, labels, opt.out, ['deps_clusters.txt', 'bow2_clusters.txt', 'bow5_clusters.txt'])


if __name__ == '__main__':
    opt = Settings.args
    main(opt)
