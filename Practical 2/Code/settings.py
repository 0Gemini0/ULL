"""
Command line settings.
"""

from argparse import ArgumentParser


def parse_settings():
    parser = ArgumentParser('User controlled settings.')

    # TODO: Paths

    # TODO: General

    # TODO: SkipGram

    # TODO: Bayesian

    # TODO: EmbedAlign

    return parser.parse_args()
