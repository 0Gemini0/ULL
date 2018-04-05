"""
This file will contain the main script.
"""
import numpy as np

from helpers import load_embeddings


def main():
    # Load word embeddings
    deps = load_embeddings("Embeddings/deps.words")
    bow2 = load_embeddings("Embeddings/bow2.words")
    bow5 = load_embeddings("Embeddings/bow5.words")


if __name__ == '__main__':
    main()
