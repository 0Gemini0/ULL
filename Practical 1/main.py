"""
This file will contain the main script.
"""
import numpy as np

from helpers import load_embeddings, get_cosine, get_correlation, retrieve_SIMLEX999_data_dict, retrieve_MEN_data_dict


def main():
    # Load word embeddings into dictionaries
    deps = load_embeddings("Embeddings/deps.words")
    bow2 = load_embeddings("Embeddings/bow2.words")
    bow5 = load_embeddings("Embeddings/bow5.words")

    # Load similarity dataset into dictionaries
    simlex = retrieve_SIMLEX999_data_dict("Similarities/SimLex-999.txt")
    men = retrieve_MEN_data_dict('Similarities/MEN_dataset_natural_form_full')

    # # Test with cosine, pearson, spearman
    # pairs = ['dog-dog', 'dog-cat', 'cat-hamster', 'turtle-tortoise']
    # cos_deps = []
    # cos_bow2 = []
    # for pair in pairs:
    #     pair = pair.split('-')
    #     cos_deps.append(get_cosine(pair[0], pair[1], deps))
    #     cos_bow2.append(get_cosine(pair[0], pair[1], bow2))
    # pearson, spearman = get_correlation(np.array(cos_deps), np.array(cos_bow2))
    #
    # print(pairs)
    # print("deps: {}".format(cos_deps))
    # print("bow2: {}".format(cos_bow2))
    # print("pearson: {}, spearman: {}".format(pearson, spearman))

    # Test with SimLex
    cos_deps = []
    simlex_scores = []
    for pair, value in simlex.items():
        pair = pair.split("_")
        if pair[0] in deps and pair[1] in deps:
            cos_deps.append(get_cosine(pair[0], pair[1], deps))
            simlex_scores.append(value)
        else:
            pass
    pearson, spearman = get_correlation(np.array(cos_deps), np.array(simlex_scores))

    print("pearson: {}, spearman: {} for SimLex".format(pearson, spearman))


if __name__ == '__main__':
    main()
