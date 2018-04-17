"""
A file for all our (helper) functions.
"""
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import os.path as osp


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


def compute_correlations(embeddings, dataset, N):
    """
    Get the correlations between a number of embedding dictionaries and a similarity dataset. The function also returns the top-N cosine scores per embedding set for pairs in the given dataset.
    """
    cosines = [[] for i in range(len(embeddings))]
    scores = []
    pearson = []
    spearman = []
    pairs = []
    top = []
    for pair, value in dataset.items():
        pair_list = pair.split("_")
        cont = False
        for emb in embeddings:
            if pair_list[0] not in emb or pair_list[1] not in emb:
                cont = True
        if cont:
            continue

        for i, emb in enumerate(embeddings):
            cosines[i].append(get_cosine(pair_list[0], pair_list[1], emb))

        # Not quite liking this dictionary unpack, but we need to track the order
        pairs.append(pair)
        scores.append(value)

    for cosine in cosines:
        top.append(sorted(zip(cosine, pairs), key=lambda cos: cos[0], reverse=True)[:N])
        p_r, s_r = get_correlation(np.array(cosine), np.array(scores))
        pearson.append(p_r[0]), spearman.append(s_r[0])

    return pearson, spearman, top


def get_cosine(word_1, word_2, embeddings):
    """
    Computes the cosine similarity of two words given an embedding dictionary.
    """
    emb_1 = embeddings[word_1]
    emb_2 = embeddings[word_2]
    return 1. - cosine(emb_1, emb_2)


def get_correlation(x, y):
    """
    Computes the spearman and pearson correlation coëfficients of two arrays.
    """
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


def retrieve_word_analogy_data_dict(path_to_data, lowercase=False):
    #################################
    """Loads the word analogy data"""
    #################################

    """Reads all the data"""
    with open(path_to_data, "r") as f:
        data_word_analogy = f.readlines()

    def l(string):
        return string.lower()

    """Creates a dict with a_a* as keys and [b b*] as values"""
    data_required = defaultdict(lambda: [])
    for line in data_word_analogy[1:]:
        if(line[0] is not ":"):
            line_contents = line.split(" ")
            if (lowercase):
                data_required[l(line_contents[0]) + "_" + l(line_contents[1])
                              ].append([l(line_contents[2]), l(line_contents[3][:-1])])
            else:
                data_required[line_contents[0] + "_" + line_contents[1]
                              ].append([line_contents[2], line_contents[3][:-1]])

    return data_required


def normalise_array(array):
    return np.divide(array, np.linalg.norm(array, 1))


def normalised_word_embeddings_data(dataset):

    normalised_dataset = {}
    word_index_map = []
    data_matrix = []

    number_of_keys = len(list(dataset.keys()))

    for i, word in enumerate(dataset):
        print("\rNormalising embedding data: " + "{:3.2f}".format(100 * (i + 1.0) / number_of_keys) +
              "% completed.", end="", flush=True)

        normalised_array = normalise_array(dataset[word])
        normalised_dataset[word] = normalised_array
        word_index_map.append(word)
        data_matrix.append(normalised_array)
    print()

    return normalised_dataset, word_index_map, np.array(data_matrix)


def create_bStars_preds_data(dataset, word_analogy_data):

    target_words = []
    bStars_preds_matrix = []

    number_of_keys = len(list(word_analogy_data.keys()))

    for j, a_aStar in enumerate(word_analogy_data):
        print("\rComputing estimated b* vectors matrix: " + "{:3.2f}".format(100 * (j + 1.0) / number_of_keys) +
              "% completed.", end="", flush=True)

        a, aStar = a_aStar.split("_")

        if (a in dataset and aStar in dataset):
            for b_bStar in word_analogy_data[a_aStar]:
                b, bStar = b_bStar

                if (b in dataset and bStar in dataset):
                    target_words.append(b_bStar)
                    bStar_pred = normalise_array(dataset[aStar] - dataset[a] + dataset[b])
                    bStars_preds_matrix.append(bStar_pred)
    print()

    return target_words, np.transpose(np.array(bStars_preds_matrix))


def compute_accuracy_and_MRR(target_pairs, word_index_map, inner_products, ignore_b=False):

    number_of_queries = float(len(target_pairs))
    correct_predictions = 0.0
    cumulative_reciprocal_rank = 0.0

    for i, target_pair in enumerate(target_pairs):
        if (ignore_b):
            print("\rComputing predictions (ignoring b): " + "{:3.2f}".format(100 * (i + 1.0) / len(target_pairs)) +
                  "% completed.", end="", flush=True)
        else:
            print("\rComputing predictions (NOT ignoring b): " + "{:3.2f}".format(100 * (i + 1.0) / len(target_pairs)) +
                  "% completed.", end="", flush=True)

        b, bStar = target_pair
        ordered_indices = np.argsort(-inner_products[:, i])

        if (ignore_b):
            if (b != word_index_map[ordered_indices[0]]):
                if (bStar == word_index_map[ordered_indices[0]]):
                    correct_predictions += 1
            else:
                if (bStar == word_index_map[ordered_indices[1]]):
                    correct_predictions += 1

            rank = 0
            for index in ordered_indices:
                if (b != word_index_map[index]):
                    rank += 1
                    if (bStar == word_index_map[index]):
                        cumulative_reciprocal_rank += 1.0/rank
                        break
        else:
            if (bStar == word_index_map[ordered_indices[0]]):
                correct_predictions += 1

            for rank, index in enumerate(ordered_indices):
                if (bStar == word_index_map[index]):
                    cumulative_reciprocal_rank += 1.0/(rank+1)
                    break
    print()
    return 100 * correct_predictions / number_of_queries, cumulative_reciprocal_rank / number_of_queries


def reduce_dimensions(embeddings, dim, mode, verbose, t_dim=50, t_num=5000):
    """
    Reduce the dimensions of word embeddings with PCA and/or TSNE.
    Args:
    * embeddings (list[np.array]): set of word embeddings.
    * dim (int): final dimension after reduction.
    * t_dim (int): if not none, embeddings are compressed to t_dim before T-SNE.
    * t_num (int): maximum number of datapoints for T-SNE.
    * verbose (bool): whether to display prints.
    * pca (bool): whether to run PCA.
    * tsne (bool): whether to run T-SNE.
    """
    clusters = []
    for embedding in embeddings:
        if mode == 'pca':
            vprint("Reducing dimensions with PCA...", verbose)
            clusters.append(reduce_dimensions_pca(embedding, dim, verbose))
        elif mode == 'tsne':
            vprint("Reducing dimensions with TSNE...", verbose)
            clusters.append(reduce_dimensions_tsne(embedding[:t_num, :], dim, t_dim, verbose))
    vprint("Done", verbose)
    return clusters


def reduce_dimensions_pca(embeddings, dim, verbose):
    pca = PCA(n_components=dim, whiten=True)
    clusters = pca.fit_transform(embeddings)
    return clusters


def reduce_dimensions_tsne(embeddings, dim, t_dim, verbose):
    tsne = TSNE(n_components=dim, verbose=verbose, n_iter=5000, perplexity=30)
    if t_dim is not None:
        pca = PCA(n_components=t_dim, whiten=True)
        pre_clusters = pca.fit_transform(embeddings)
    else:
        pre_clusters = embeddings
    clusters = tsne.fit_transform(pre_clusters)
    return clusters


def visualize_embeddings(embeddings, labels, titles, num, dim, verbose):
    """
    Visualize a list of dimension-reduced embeddings.
    """
    if not labels:
        labels = [np.ones(num)] * len(embeddings)
    for embedding, label, title in zip(embeddings, labels, titles):
        if dim == 2:
            visualize_embeddings_2d(embedding, label, title, num)
        elif dim == 3:
            visualize_embeddings_3d(embedding, label, title, num)
        else:
            vprint("Cannot visualize in {} dimensional space".format(dim), verbose)


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
    """
    Visualize 3-D compressed embeddings.
    """
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(embeddings[:num, 0], embeddings[:num, 1], embeddings[:num, 2], c=labels[:num].astype(float))
    ax.set_xlabel("dim_1")
    ax.set_ylabel("dim_2")
    ax.set_zlabel("dim_3")
    ax.set_title(title)
    plt.show()


def cluster(embeddings, mode, verbose, k=2, eps=0.1, min_samples=4):
    """
    Cluster a set of embeddings with a method of choice.
    """
    labels = []
    for i, embedding in enumerate(embeddings):
        if mode == 'distance':
            vprint("Clustering set {} with a distance metric (K-means)...".format(i), verbose)
            labels.append(cluster_distance(embedding, k))
        elif mode == 'density':
            vprint("Clustering set {} with a density metric (DBSCAN)...".format(i), verbose)
            labels.append(cluster_density(embedding, eps, min_samples))
        else:
            raise Exception("Unknown clustering method, select [distance, density]")
    vprint("Done", verbose)
    return labels


def cluster_distance(embeddings, k):
    """
    Cluster a group of nouns based on their embeddings with a distance metric.
    """
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    return labels


def cluster_density(embeddings, eps, min_samples):
    """
    Cluster a group of nouns based on their embeddings with a density metric.
    """
    dbscan = DBSCAN(min_samples=min_samples, eps=eps, metric='cosine')
    labels = dbscan.fit_predict(embeddings)
    return labels


def save_clusters(nouns, labels, path, names):
    """
    Write clusters to a file for later inspection.
    """
    if not osp.isdir(path):
        os.makedirs(path)

    for label, name in zip(labels, names):
        with open(osp.join(path, name), 'w', encoding='utf8') as f:
            for i in range(max(label) + 1):
                f.write("Cluster {}: {}\n".format(i, [nouns[j] for j in list(np.argwhere(label == i).flatten())]))


def vprint(message, verbose=True):
    if verbose:
        print(message)
