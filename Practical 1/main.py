"""
This file will contain the main script.
"""


def main():
    # Load word embeddings
    with open("Embeddings/bow2.words", 'r') as f:
        for i, line in enumerate(f):
            print(line)
            if i > 100:
                break


if __name__ == '__main__':
    main()
