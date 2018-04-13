from argparse import ArgumentParser

class Settings:
    """Wrapper for argument parser"""
    parser = ArgumentParser('Main script settings.')
    parser.add_argument('--exercise', type=int, default=0,
                        help='Which excercise to run')
    parser.add_argument('--tsne_num', type=int, default=5000,
                        help='Number of embeddings to input T-SNE.')
    parser.add_argument('--k', type=int, default=2,
                        help='Number of K-Means clusters.')
    parser.add_argument('--nouns', type=str, default="./nouns.txt",
                        help="Path to nouns.")
    parser.add_argument('--dim', type=int, default=2,
                        help='Number of dimensions to reduce embeddings to.')
    args = parser.parse_args()
