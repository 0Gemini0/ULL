#!/usr/bin/python3

"""
Data preprocessing.
"""

###########
# Imports #
###########
import numpy as np
from numpy.random import multinomial
from collections import defaultdict, Counter
import operator
import pickle
from time import time

# path_to_data = "../Data/Original Data/test_data_file.txt"
path_to_data = "../Data/Original Data/hansards/training.en"
# path_to_data = "../Data/Original Data/europarl/training.en"

#####################################
# Data loading/processing functions #
#####################################


class MutableInt(object):

    def __init__(self, initial_value):
        self._integer = initial_value

    def update_integer(self, new_value):
        self._integer = new_value

    def increment_value(self):
        self._integer += 1

    @property
    def integer(self):
        return self._integer


def line_mutate(line, lowercase, dictionary, index_word_map=None, index=None, counter=None, dict_operation="Word Count"):
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
        counter.update(line)
        for word in line:
            if (word not in dictionary):
                dictionary[word] = index.integer
                index_word_map.append(word)
                index.increment_value()

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


def preprocess_data_skipgram(path_to_data, window_size, k=1, lowercase=True, store_sequentially=False):
    ###############################################################################
    """Loads the english side of the dataset corresponding to the provided path."""
    ###############################################################################

    '''Load the data.'''
    with open(path_to_data, "r", encoding='utf-8') as f:
        data_lines = f.readlines()

    '''Apply line_mutate to all lines in the dataset.'''
    word_index_map = {}
    index_word_map = []
    index = MutableInt(0)
    counter = Counter()
    data_lines = [line_mutate(line, lowercase, word_index_map, index_word_map, index, counter, "Word Index")
                  for line in data_lines]

    '''Compute unigram statistics of corpus.'''
    N = float(sum(counter.values()))
    unigram_statistics = {k: v/N for k, v in counter.items()}
    ordered_unigram_statistics = [0]*len(index_word_map)
    for word in unigram_statistics:
        ordered_unigram_statistics[word_index_map[word]] = unigram_statistics[word]

    '''Extra function for helping compute negative samples.'''
    def get_samples_from_multinomial(counts):
        samples = [index for index in np.flatnonzero(counts) for _ in range(counts[index])]
        # for index in np.flatnonzero(counts):
        #     for _ in range(counts[index]):
        #         samples.append(index)
        return samples

    """Compute context (past and future) and negative samples for each token in the dataset."""
    centre_word_context_windows = []
    negative_samples = []
    a = len(data_lines)
    for m, line in enumerate(data_lines):
        # print('{:2f}'.format(100*(m+1)/a), end="\n", flush=True)
        '''Compute context (past and future).'''
        append_time = 0.0
        sample_time = 0.0
        for i, word in enumerate(line):
            past_context = []
            future_context = []
            for j in range(max(0, i - window_size), i):  # Compute past context
                past_context.append(word_index_map[line[j]])
            for j in range(i + 1, min(i + 1 + window_size, len(line))):  # Compute future context
                future_context.append(word_index_map[line[j]])
            centre_word_context_windows.append((word_index_map[word], [past_context, future_context]))

            length_context = len(past_context) + len(future_context)

            '''Compute negative samples.'''
            samples = get_samples_from_multinomial(multinomial(k * length_context, ordered_unigram_statistics))
            negative_samples.append((word_index_map[word], samples))

        print('\rPercentage done: {:2f}'.format(100*(m+1)/a), end='', flush=True)

    """Write data to files."""
    '''If saving sequentially.'''  # TODO: compute location of each example.
    if (store_sequentially):
        '''Positive samples.'''
        with open(path_to_data[0:-3] + "_" + str(k) + "_" + str(lowercase) + "samples" + path_to_data[-3:], "w", encoding='utf-8') as f:
            for context_window in centre_word_context_windows:
                pickle.dump(context_window, f)

        '''Negative samples.'''
        with open(path_to_data[0:-3] + "_" + str(k) + "_" + str(lowercase) + "samples" + path_to_data[-3:], "w", encoding='utf-8') as f:
            for negative_sample in negative_samples:
                pickle.dump(negative_sample, f)

    # '''If all examples at once.''' TODO: FIX STUPID COMMENT
    else:
        '''Positive samples.'''
        pickle.dump(centre_word_context_windows, open(
            path_to_data[0:-3] + "_" + str(k) + "_" + str(lowercase) + "samples" + path_to_data[-3:], "wb"))

        '''Negative samples.'''
        pickle.dump(negative_samples, open(path_to_data[0:-3] + "_" + str(k) +
                                           "_" + str(lowercase) + "negativeSamples" + path_to_data[-3:], "wb"))

    '''Word Index Map.'''
    pickle.dump(word_index_map, open(path_to_data[0:-3] + "_" + str(k) +
                                     "_" + str(lowercase) + "wordIndexMap" + path_to_data[-3:], "wb"))

    '''Index Word Map.'''
    pickle.dump(index_word_map, open(path_to_data[0:-3] + "_" + str(k) +
                                     "_" + str(lowercase) + "indexWordMap" + path_to_data[-3:], "wb"))


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


if __name__ == '__main__':
    basic_dataset_preprocess(path_to_data)

    preprocess_data_skipgram(path_to_data[:-3] + '_10000_True.en', 2)
