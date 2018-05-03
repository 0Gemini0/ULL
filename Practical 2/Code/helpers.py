"""
Helper functions.
"""

###########
# Imports #
###########
import numpy as np
from collections import defaultdict
import operator

path_to_data = "../Data/Original Data/test_data_file.txt"
# path_to_data = "../Data/Original Data/hansards/training.en"
# path_to_data = "../Data/Original Data/europarl/training.en"

#####################################
# Data loading/processing functions #
#####################################


def line_mutate(line, lowercase, dictionary, index_word_map=None, index=None, dict_operation="Word Count"):
    ##############################################################################################################
    """Function to lowercase all words and strip '\r' and '\n' symbols. Also, implicitly counts word frequency."""
    ##############################################################################################################

    '''Split the line on spaces and remove EOL characters from last word.'''
    line = line.split(" ")
    line[-1] = line[-1].rstrip("\n").rstrip("\r")

    '''Lowercase(?) all words in line.'''
    if (lowercase):
        line = [word.lower() for word in line]

    '''Compute counts for all words in line.'''
    if (dict_operation == "Word Count"):
        for word in line:
            dictionary[word] += 1

    '''Indexes all words in the line (implicitly, in the dataset).'''
    if (dict_operation == "Word Index"):
        for word in line:
            if (word not in dictionary):
                dictionary[word] = index
                index_word_map.append(word)
                index += 1

    return line


def basic_dataset_preprocess(path_to_data, threshold=10000, lowercase=True):
    ############################################################################
    """Keep 'threshold' most frequent words. Lowercase(?) them. UNK the rest."""
    ############################################################################

    '''Load the data.'''
    with open(path_to_data, "r", encoding='utf-8') as f:
        data_lines = f.readlines()

    '''Apply line mutate to all lines in the dataset.'''
    word_count = defaultdict(lambda: 0)
    data_lines = [line_mutate(line, lowercase, word_count) for line in data_lines]

    '''Order words on their frequency.'''
    ordered_counts = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)

    print(ordered_counts)

    '''Specify which words will UNKed and which won't.'''
    to_UNK_dict = {}
    for i, pair in enumerate(ordered_counts):
        if (i <= threshold):
            to_UNK_dict[pair[0]] = pair[0]
        else:
            to_UNK_dict[pair[0]] = "UNK"

    '''Save the UNKed (on threshold) and lowercased(?) dataset to file.'''
    with open(path_to_data[0:-3] + "_" + str(threshold) + "_" + str(lowercase) + path_to_data[-3:], "w", encoding='utf-8') as f:
        for line in data_lines:
            new_line = ""
            for word in line:
                new_line += to_UNK_dict[word] + " "
            f.write(new_line[0:-1] + "\n")

# TODO: FIX THIS FUNCTION!!!!!!!!!! CURRENTLY NOT WORKING!
def preprocess_data_skipgram(path_to_data, window_size, k=1, lowercase=True, store_sequentially=False):
    ###############################################################################
    """Loads the english side of the dataset corresponding to the provided path."""
    ###############################################################################

    '''Load the data.'''
    with open(path_to_data, "r") as f:  # TODO: Add ', encoding='utf-8''
        data_lines = f.readlines()

    '''Apply line_mutate to all lines in the dataset.'''
    word_index_map = {}
    index_word_map = []
    index = 0
    data_lines = [line_mutate(line, lowercase, word_index_map, index_word_map, index, "Word Index") for line in data_lines]

    '''Compute context (past and future) for each token in the dataset.'''
    centre_word_context_windows = defaultdict(lambda: [])  # TODO: Convert into list.
    for line in data_lines:
        for i, word in enumerate(line):
            past_context = []
            future_context = []
            for j in range(max(0, i - window_size), i):  # Compute past context
                past_context.append(word_index_map[line[j]])
            for j in range(i + 1, min(i + 1 + window_size, len(line))):  # Compute future context
                future_context.append(word_index_map[line[j]])
            centre_word_context_windows[word_index_map[word]].append([past_context, future_context])  # Add past/future context to word

    print(centre_word_context_windows)
    print(word_index_map)
    print(index_word_map)
preprocess_data_skipgram(path_to_data, 2)

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
