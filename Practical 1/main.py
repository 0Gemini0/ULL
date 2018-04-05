"""
This file will contain the main script.
"""
import numpy as np

from helpers import load_embeddings, get_cosine


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


if __name__ == '__main__':
    main()
