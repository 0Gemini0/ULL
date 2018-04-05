"""
A file for all our (helper) functions.
"""
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr


def load_embeddings(path):
    """Loads a file of word embeddings stored with a single embedding per line"""
    print("Loading {}...".format(path.split("/")[-1]))
    words = dict()
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            word = line.split()
            words[word[0]] = np.array(word[1:]).astype(np.float32)
    print("Done")
    return words


def get_cosine(word_1, word_2, embeddings):
    """Computes the cosine similarity of two words given an embedding dictionary."""
    emb_1 = embeddings[word_1]
    emb_2 = embeddings[word_2]
    return 1. - cosine(emb_1, emb_2)


def get_correlation(x, y):
    """Computes the spearman and pearson correlation coëfficients of two arrays"""
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return pearson, spearman
