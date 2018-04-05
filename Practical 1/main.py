"""
This file will contain the main script.
"""
import numpy as np

from helpers import load_embeddings, get_cosine, get_correlation


def main():
    # Load word embeddings
    deps = load_embeddings("Embeddings/deps.words")
    bow2 = load_embeddings("Embeddings/bow2.words")
    bow5 = load_embeddings("Embeddings/bow5.words")

    # Test with cosine
    print(get_cosine('cat', 'dog', deps))
    print(get_cosine('cat', 'dog', bow2))
    print(get_cosine('cat', 'dog', bow5))

    print(get_cosine('dog', 'dog', deps))
    print(get_cosine('dog', 'dog', bow2))
    print(get_cosine('dog', 'dog', bow5))

    # Test with pearson, spearman
    pairs = ['dog-dog', 'dog-cat', 'cat-hamster', 'turtle-tortoise']
    cos_deps = []
    cos_bow2 = []
    for pair in pairs:
        pair = pair.split('-')
        cos_deps.append(get_cosine(pair[0], pair[1], deps))
        cos_bow2.append(get_cosine(pair[0], pair[1], bow2))
    pearson, spearman = get_correlation(np.array(cos_deps), np.array(cos_bow2))

    print(pairs)
    print("deps: {}".format(cos_deps))
    print("bow2: {}".format(cos_bow2))
    print("pearson: {}, spearman: {}".format(pearson, spearman))


if __name__ == '__main__':
    main()
