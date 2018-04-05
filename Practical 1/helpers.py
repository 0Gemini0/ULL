"""
A file for all our (helper) functions.
"""
import numpy as np


def load_embeddings(path):
    print("Loading {}...".format(path.split("/")[-1]))
    words = dict()
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            word = line.split()
            words[word[0]] = np.array(word[1:]).astype(np.float32)
    print("Done")
    return words
