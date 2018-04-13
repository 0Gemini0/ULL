"""
A file for all our (helper) functions.
"""
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_embeddings(path, verbose=True):
    """Loads a file of word embeddings stored with a single embedding per line"""
    vprint("Loading {}...".format(path.split("/")[-1]), verbose)
    words = dict()
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            word = line.split()
            words[word[0]] = np.array(word[1:]).astype(np.float32)
    vprint("Done", verbose)
    return words


def compute_correlations(embeddings, dataset):
    """Get the correlations between a number of embedding dictionaries and a similarity dataset"""
    cosines = [[] for i in range(len(embeddings))]
    scores = []
    pearson = []
    spearman = []
    for pair, value in dataset.items():
        pair = pair.split("_")
        cont = False
        for emb in embeddings:
            if pair[0] not in emb or pair[1] not in emb:
                cont = True
        if cont:
            continue

        for i, emb in enumerate(embeddings):
            cosines[i].append(get_cosine(pair[0], pair[1], emb))

        scores.append(value)

    for cosine in cosines:
        p_r, s_r = get_correlation(np.array(cosine), np.array(scores))
        pearson.append(p_r), spearman.append(s_r)

    return pearson, spearman


def get_cosine(word_1, word_2, embeddings):
    """Computes the cosine similarity of two words given an embedding dictionary."""
    emb_1 = embeddings[word_1]
    emb_2 = embeddings[word_2]
    return 1. - cosine(emb_1, emb_2)


def get_correlation(x, y):
    """Computes the spearman and pearson correlation coëfficients of two arrays"""
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return pearson, spearman


def retrieve_SIMLEX999_data_dict(path_to_data):
    ########################################################
    """Loads the relevant SimLex-999 for the task at hand"""
    ########################################################

    """Reads all the data"""
    with open(path_to_data, "r") as f:
        data_simlex = f.readlines()

    """Creates a dict with word pairs as keys and the SimLex-999 score as value"""
    data_required = {}
    for line in data_simlex[1:]:
        line_contents = line.split("\t")
        data_required[line_contents[0] + "_" + line_contents[1]] = float(line_contents[3])

    return data_required


def retrieve_MEN_data_dict(path_to_data):
    ########################
    """Loads the MEN data"""
    ########################

    """Reads all the data"""
    with open(path_to_data, "r") as f:
        data_men = f.readlines()

    """Creates a dict with word pairs as keys and the MEN score as value"""
    data_required = {}
    for line in data_men:
        line_contents = line.split(" ")
        data_required[line_contents[0] + "_" + line_contents[1]] = float(line_contents[2][:-1])

    return data_required


def retrieve_word_analogy_data_dict (path_to_data):
    #################################
    """Loads the word analogy data"""
    #################################

    """Reads all the data"""
    with open(path_to_data, "r") as f:
        data_word_analogy = f.readlines()

    """Creates a dict with a_a* as keys and [b b*] as values"""
    data_required = defaultdict(lambda: [])
    for line in data_word_analogy[1:]:
        if(line[0] is not ":"):
            line_contents = line.split(" ")
            data_required[line_contents[0] + "_" + line_contents[1]].append([line_contents[2], line_contents[3][:-1]])

    return data_required


def reduce_dimensions(embeddings, dim, t_dim=None, t_num=5000, verbose=True):
    """
    Reduce the dimensions of word embeddings with PCA and TSNE
    Args:
    * embeddings (dict): word embeddings with keys.
    * dim (int): final dimension after reduction.
    * t_dim (int): if not none, embeddings are compressed to t_dim before tsne.
    """

    # Define dimension reduction algorithms
    pca = PCA(n_components=dim)
    pca_tsne = PCA(n_components=t_dim)
    tsne = TSNE(n_components=dim, verbose=verbose)

    # Reduce dimensions with PCA and TSNE
    vprint("Reducing dimensions with PCA...", verbose)
    pca_clusters = pca.fit_transform(embeddings)
    if t_dim is not None:
        vprint("Pre-reducing dimensions for TSNE...", verbose)
        tsne_pre_clusters = pca_tsne.fit_transform(embeddings)
    else:
        tsne_pre_clusters = embeddings
    vprint("Reducing dimensions with TSNE...", verbose)
    tsne_clusters = tsne.fit_transform(tsne_pre_clusters[:t_num, :])
    vprint("Done", verbose)

    return pca_clusters, tsne_clusters


def visualize_embeddings_2d(embeddings, labels, title, num):
    """
    Visualize 2-D compressed embeddings.
    """
    plt.scatter(embeddings[:num, 0], embeddings[:num, 1], c=labels[:num].astype(float))
    plt.xlabel("dim_1")
    plt.ylabel("dim_2")
    plt.title(title)
    plt.show()


def visualize_embeddings_3d(embeddings, labels, title, num):
    fig = plt.figure(figsize=(4,3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(embeddings[:num, 0], embeddings[:num, 1], embeddings[:num, 2], c=labels[:num].astype(float))
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.set_zlabel("dim_3")
    ax.set_title(title)

    plt.show()


def cluster(embeddings, k):
    """
    Cluster a group of nouns based on their embeddings.
    """
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    return labels


def vprint(message, verbose=True):
    if verbose:
        print(message)
