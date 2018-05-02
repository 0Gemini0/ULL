"""
Command line settings.
"""

from argparse import ArgumentParser


def parse_settings():
    parser = ArgumentParser('User controlled settings.')

    # Paths
    parser.add_argument('--out_path', type=str, default="../Out",
                        help="Output folder.")
    parser.add_argument('--data_path', type=str, default='../Data',
                        help="Path to folder containing datasets.")

    # TODO: General

    # TODO: SkipGram

    # TODO: Bayesian

    # TODO: EmbedAlign

    return parser.parse_args()
