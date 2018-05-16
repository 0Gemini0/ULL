"""
Command line settings.
"""

from argparse import ArgumentParser


def parse_settings():
    parser = ArgumentParser('User controlled settings.')

    # Paths
    parser.add_argument('--out_path', type=str, default="../out",
                        help="Output folder.")
    parser.add_argument('--data_path', type=str, default='../data/original',
                        help="Path to folder containing datasets.")
    parser.add_argument('--lst_path', type=str, default='../lst',
                        help="Lst path.")
    parser.add_argument('--aer_path', type=str, default='../wa')
    parser.add_argument('--dataset', type=str, default='hansards',
                        help='Which dataset to load, choose [hansards, europarl]')
    parser.add_argument('--training_test', type=str, default='training',
                        help='Which dataset to load, choose [training, test]')
    parser.add_argument('--language', type=str, default='.en',
                        help='Language to preprocess. Choose [.en, .fr]')

    # Preprocessing
    parser.add_argument('--vocab_size', type=int, default=0,
                        help="Size of vocalubary.")
    parser.add_argument('--lowercase', type=int, default=1,
                        help="Whether the dataset is/should be lowercased. Choose [0, 1]")
    parser.add_argument('--k', type=int, default=1,
                        help="Amount of negative samples per positive sample.")
    parser.add_argument('--window_size', type=int, default=5,
                        help="Size of context window around central word.")
    parser.add_argument('--max_sentence_size', type=int, default=30,
                        help='Maximum sentence size (int).')
    parser.add_argument('--save_sequential', type=int, default=0,
                        help='Whether to save sequentially. Choose [0, 1].')

    # Model architecture
    parser.add_argument('--model', type=str, default='skipgram',
                        help='Which model to use. Choose [skipgram, bayesian, embedalign].')
    parser.add_argument('--v_dim_en', type=int, default=10002,
                        help='Dimensionality of input layer V.')
    parser.add_argument('--v_dim_fr', type=int, default=10002,
                        help='Dimensionality of input layer V.')
    parser.add_argument('--d_dim', type=int, default=100,
                        help='Dimensionality of embedding layer D.')
    parser.add_argument('--h_dim', type=int, default=128,
                        help='Dimensionality of hidden layers H.')
    parser.add_argument('--neg_dim', type=int, default=2500)

    # Training
    parser.add_argument('--parallel', type=int, default=0,
                        help='Whether to use parallel processing if available.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs over the data.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Number of datapoints per minibatch.")
    parser.add_argument('--kl_step', type=int, default=1e-3,
                        help="Kl annealing step.")

    # Evaluation
    parser.add_argument('--aer_mode', type=str, default='test',
                        help="Alignment error rate prediction mode, choose [dev, test]")

    return parser.parse_args()
