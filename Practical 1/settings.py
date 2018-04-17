from argparse import ArgumentParser


class Settings:
    """Wrapper for argument parser"""
    parser = ArgumentParser('Main script settings.')
    # Paths
    parser.add_argument('--out_path', type=str, default="./Out",
                        help="Output folder.")
    parser.add_argument('--data_path', type=str, default='./Data',
                        help="Path to folder containing datasets.")
    parser.add_argument('--emb_path', type=str, default='./Embeddings',
                        help="Path to embedding folder.")

    # General
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity of printing, 0=None, 1=Full')
    parser.add_argument('--exercise', type=int, default=1,
                        help='Which excercise to run. Choose 3, 4, 5 or 1 for all.')

    # Excercise 3
    parser.add_argument('--N', type=int, default=20,
                        help='Display top N most cosine word pairs on task 3.')

    # Excercise 4

    # Excercise 5
    parser.add_argument('--tsne_num', type=int, default=5000,
                        help='Max number of embeddings to input T-SNE.')
    parser.add_argument('--tsne_dim', type=int, default=50,
                        help='Pre-reduce embeddings to this dimension before T-SNE.')
    parser.add_argument('--k', type=int, default=30,
                        help='Number of K-Means clusters.')
    parser.add_argument('--eps', type=float, default=0.3,
                        help='eps parameter for DBSCAN.')
    parser.add_argument('--min_samples', type=int, default=2,
                        help='min_samples parameter for DBSCAN.')
    parser.add_argument('--red_mode', type=str, default='tsne',
                        help="Type of dimensionality reduction, [pca, tsne]")
    parser.add_argument('--clu_mode', type=str, default='density',
                        help='Type of clustering, [density, distance]')
    parser.add_argument('--dim', type=int, default=2,
                        help='Number of dimensions to reduce embeddings to.')
    parser.add_argument('--viz_num', type=int, default=5000,
                        help='Maximum number of embedding points to visualize.')

    args = parser.parse_args()
