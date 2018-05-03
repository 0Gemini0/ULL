"""
Helper functions.
"""

###########
# Imports #
###########
import numpy as np
from collections import defaultdict
import operator
import pickle
# path_to_data = "../Data/Original Data/test_data_file.txt"
# path_to_data = "../Data/Original Data/hansards/training.en"
path_to_data = "../Data/Original Data/europarl/training.en"

#####################################
# Data loading/processing functions #
#####################################

def basic_dataset_preprocess(path_to_data, threshold=10000, lowercase=True):
    ############################################################################
    """Keep 'threshold' most frequent words. Lowercase(?) them. UNK the rest."""
    ############################################################################

    '''Load the data.'''
    with open(path_to_data, "r", encoding='utf-8') as f:
        data_lines = f.readlines()

    def line_mutate(line):
        line = line.split(" ")
        line[-1] = line[-1].rstrip("\n").rstrip("\r")

        if (lowercase):
            line = [word.lower() for word in line]

        for word in line:
            word_count[word] += 1

        return line

    word_count = defaultdict(lambda: 0)
    data_lines = [line_mutate(line) for line in data_lines]

    ordered_counts = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

    to_UNK_dict = {}
    for i, pair in enumerate(ordered_counts):
        if (i <= threshold):
            to_UNK_dict[pair[0]] = pair[0]
        else:
            to_UNK_dict[pair[0]] = "UNK"

    with open(path_to_data[0:-3] + "_" + str(threshold) + "_" + str(lowercase) + path_to_data[-3:], "w", encoding='utf-8') as f:
        for line in data_lines:
            new_line = ""
            for word in line:
                new_line += to_UNK_dict[word] + " "
            f.write(new_line[0:-1] + "\n")



basic_dataset_preprocess(path_to_data)



def preprocess_data_skipgram(path_to_data, window_size, k=1, store_sequentially=False):
    ###############################################################################
    """Loads the english side of the dataset corresponding to the provided path."""
    ###############################################################################

    '''Load the data.'''
    with open(path_to_data, "r") as f:
        data_lines = f.readlines()

    observed_pairs = []
    index = 0
    word_index_map = {}
    index_word_map = []

    '''Register observed pairs and word/index index/word maps.'''
    for line in data_lines:  # TODO: add check for stupid lines? Like " * * * ".
        line_list = line.split(" ")
        line_list[-1] = line_list[-1].rstrip("\n").rstrip("\r")

        for i in range(len(line_list) - 1):
            if (line_list[i] not in word_index_map):
                word_index_map[line_list[i]] = index
                index_word_map.append(line_list[i])
                index += 1

            for j in range(i+1, len(line_list)):
                observed_pairs.append([line_list[i], line_list[j]])
                observed_pairs.append([line_list[j], line_list[i]])

        if (line_list[len(line_list) - 1] not in word_index_map):
            word_index_map[line_list[len(line_list) - 1]] = index
            index_word_map.append(line_list[len(line_list) - 1])
            index += 1

    print("Observed Pairs: ", observed_pairs)
    print("Word Index Map: ", word_index_map)
    print("Index Word Map: ", index_word_map)
    '''Generate negative pairs.'''
    negative_pairs = []
    for word in word_index_map:
        random_context_words_indices = np.random.randint(0, len(index_word_map), k)
        for i in random_context_words_indices:
            negative_pairs.append([word, index_word_map[i]])
    print("Negative Pairs: ", negative_pairs)

    '''Save pre-processed data.'''


# a = "ajsh\n"
# b = "ajsh\r\n"
# c = "ajsh\r"
# d = "ajsh"
#
# print(d.rstrip("\n").rstrip("\r"))
# print(repr(a))
# a = a.rstrip("\n")
# print(repr(a))
# a = a.rstrip("\r")
# print(repr(a))


# preprocess_data_skipgram(path_to_data, 2)



def damned_experimental_subsampler():
    '''This is an experimental "dirty" (as per Omar Levy) subsampler.'''
    '''It appears to get very similar dropping probabilities for corpora with different sizes.'''
    # ordered_counts = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    # max_count = ordered_counts[0][1]
    #
    # unigram_probablities = {}
    # for word in word_count:
    #     unigram_probablities[word] = word_count[word] / total_number_of_words
    #
    # ordered_unigram_probabilities = sorted(unigram_probablities.items(), key=operator.itemgetter(1), reverse=True)
    #
    # total_count_most_frequent_words = 0
    # words_above_threshold = []
    # for unigram_probability in ordered_unigram_probabilities:
    #     if (unigram_probability[1] >= threshold):
    #         total_count_most_frequent_words += word_count[unigram_probability[0]]
    #         words_above_threshold.append(unigram_probability[0])
    #         min_count = word_count[unigram_probability[0]]
    #     else:
    #         break
    #
    # word_drop_out_prob = {}
    # for word in word_count:
    #     if (word in words_above_threshold):  # TODO: Maybe just set it to 1?
    #         word_drop_out_prob[word] = (word_count[word] / (
    #         max_count * (1 + min_count / total_count_most_frequent_words))) ** (
    #                                    1 / 10)  # TODO: Find an appropriate value
    #     else:
    #         word_drop_out_prob[word] = 0
    #
    # i = 0
    # flag = False
    # for pair in ordered_counts:
    #     print("Word: " + pair[0] + " ||| Count: " + str(word_count[pair[0]]) + " ||| Dropout prob: " + str(
    #         word_drop_out_prob[pair[0]]))
    #     if (word_drop_out_prob[pair[0]] >= 0.2):
    #         flag = True
    #     if (flag):
    #         i += 1
    #         if (i > 20):
    #             break
    raise NotImplementedError()
