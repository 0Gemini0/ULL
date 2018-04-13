from argparse import ArgumentParser

class Settings:
    """Wrapper for argument parser"""
    parser = ArgumentParser('Main script settings.')
    parser.add_argument('--exercise', type=int, default=0,
                        help='Which excercise to run')
    args = parser.parse_args()
