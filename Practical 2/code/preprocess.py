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
import os.path as osp
import msgpack
from settings import parse_settings

# Patch msgpack to work with numpy
import msgpack_numpy as m
m.patch()


# path_to_data = "../data/original/test_data_file.tx"

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


def line_mutate(line, lowercase, dictionary, index_word_map=None, index=None, counter=None,
                dict_operation="Word Count"):
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
        if (i < threshold):
            to_UNK_dict[pair[0]] = pair[0]
        else:
            to_UNK_dict[pair[0]] = "UNK"

    '''Save the UNKed (on threshold) and lowercased(?) dataset to file.'''
    filename = path_to_data[0:-3] + "_" + str(threshold) + "_" + str(bool(lowercase)) + path_to_data[-3:]
    with open(filename, "w", encoding='utf-8') as f:
        for line in data_lines:
            new_line = ""
            for word in line:
                new_line += to_UNK_dict[word] + " "
            f.write(new_line[0:-1] + "\n")

    return filename


def preprocess_data_skipgram(path_to_data, window_size, pad_index, k=1, store_sequentially=False):
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
    data_lines = [line_mutate(line, False, word_index_map, index_word_map, index, counter, "Word Index")
                  for line in data_lines]

    '''Compute unigram statistics of corpus.'''
    N = float(sum(counter.values()))
    unigram_statistics = {k: v/N for k, v in counter.items()}
    ordered_unigram_statistics = [0]*len(index_word_map)
    for word in unigram_statistics:
        ordered_unigram_statistics[word_index_map[word]] = unigram_statistics[word]

    '''Extra function for helping compute negative samples.'''
    def get_samples_from_multinomial(counts):
        samples = [int(index) for index in np.flatnonzero(counts) for _ in range(counts[index])]
        return samples

    """Compute context (past and future) and negative samples for each token in the dataset."""
    centre_word_context_windows = []
    negative_samples = []
    a = len(data_lines)
    for m, line in enumerate(data_lines):
        for i, word in enumerate(line):
            '''Compute context (past and future).'''
            past_context = []
            future_context = []
            for j in range(max(0, i - window_size), i):  # Compute past context
                past_context.append(word_index_map[line[j]])
            for j in range(i + 1, min(i + 1 + window_size, len(line))):  # Compute future context
                future_context.append(word_index_map[line[j]])

            '''Compute negative samples.'''
            length_context = len(past_context) + len(future_context)
            samples = get_samples_from_multinomial(multinomial(k * length_context, ordered_unigram_statistics))

            '''Pad context windows to full size.'''
            if len(past_context) < window_size:
                pad = [pad_index] * (window_size - len(past_context))
                past_context = pad + past_context
                samples = pad + samples
            if len(future_context) < window_size:
                pad = [pad_index] * (window_size - len(future_context))
                future_context.extend(pad)
                samples.extend(pad)

            '''Store.'''
            centre_word_context_windows.append((word_index_map[word], [past_context, future_context]))
            negative_samples.append((word_index_map[word], samples))

        print('\rPercentage done: {:2f}'.format(100*(m+1)/a), end='', flush=True)

    """Write data to files."""
    '''If saving sequentially.'''  # TODO: compute location of each example.
    if (store_sequentially):
        # TODO: seq storing with msgpack
        raise NotImplementedError()
        # '''Positive samples.'''
        # with open(path_to_data[0:-3] + "_" + str(window_size) + '_' + str(k) + "_" + "samples" + path_to_data[-3:], "w", encoding='utf-8') as f:
        #     for context_window in centre_word_context_windows:
        #         pickle.dump(context_window, f)
        #
        # '''Negative samples.'''
        # with open(path_to_data[0:-3] + "_" + str(window_size) + '_' + str(k) + "_" + "samples" + path_to_data[-3:], "w", encoding='utf-8') as f:
        #     for negative_sample in negative_samples:
        #         pickle.dump(negative_sample, f)

    # '''If all examples at once.''' TODO: FIX STUPID COMMENT
    else:
        '''Positive samples.'''
        msgpack.dump(centre_word_context_windows, open(
            path_to_data[0:-3] + "_" + str(window_size) + '_' + str(k) + "_" + "samples" + path_to_data[-3:], "wb"))

        '''Negative samples.'''
        msgpack.dump(negative_samples, open(path_to_data[0:-3] + "_" + str(window_size) + '_' +
                                            str(k) + "_" + "negativeSamples" + path_to_data[-3:], "wb"))

    '''Word Index Map. use_bin_type=True to pack strings, paired with unpacking with encoding=utf-8'''
    msgpack.dump(word_index_map, open(path_to_data[0:-3] + "_" + str(window_size) + '_' +
                                      str(k) + "_" + "wordIndexMap" + path_to_data[-3:], "wb"), use_bin_type=True)

    '''Index Word Map.'''
    msgpack.dump(index_word_map, open(path_to_data[0:-3] + "_" + str(window_size) + '_' +
                                      str(k) + "_" + "indexWordMap" + path_to_data[-3:], "wb"), use_bin_type=True)


# Assumes filenames will be "training.en" and "training.fr"
def preprocess_data_embedalign(path_to_data, training_test, lowercase, max_sentence_size, threshold):
    ###############################################################################
    """Loads the english side of the dataset corresponding to the provided path."""
    ###############################################################################

    '''Load the data.'''
    print("Loading data...")
    with open(path_to_data + training_test + ".en", "r", encoding='utf-8') as f:
        data_lines_en = f.readlines()
    with open(path_to_data + training_test + ".fr", "r", encoding='utf-8') as f:
        data_lines_fr = f.readlines()
    print("Loaded data.\n")

    print("Getting sentences' sizes...")
    '''Get original dataset sentences' lengths'''
    lengths_sentences_en = [len(sentence.split(" ")) for sentence in data_lines_en]
    lengths_sentences_fr = [len(sentence.split(" ")) for sentence in data_lines_fr]
    print("Got sentences' sizes.\n")

    def basic_line_mutate_vocab_size(line, lowercase, counter=None):
        '''Split the line on spaces and remove EOL characters from last word.'''
        line = line.split(" ")
        line[-1] = line[-1].rstrip("\n").rstrip("\r")
        if (line[-1] == ""):
            line = line[:-1]

        '''Lowercase(?) all words in line.'''
        if (lowercase):
            line = [word.lower() for word in line]

        if (counter is not None):
            counter.update(line)

        return line

    counter_en = Counter()
    counter_fr = Counter()

    print("Counting words and basic preprocessing...")
    data_lines_en = [basic_line_mutate_vocab_size(line, lowercase, counter_en) for i, line in enumerate(data_lines_en)
                     if (lengths_sentences_en[i] <= max_sentence_size and lengths_sentences_fr[i] <= max_sentence_size)]
    data_lines_fr = [basic_line_mutate_vocab_size(line, lowercase, counter_fr) for i, line in enumerate(data_lines_fr)
                     if (lengths_sentences_en[i] <= max_sentence_size and lengths_sentences_fr[i] <= max_sentence_size)]
    print("Counted words and performed basic preprocessing.\n")

    pad_index_en = threshold + 1
    pad_index_fr = threshold + 1
    if (threshold == 0):
        pad_index_en = len(list(counter_en.keys()))
        pad_index_fr = len(list(counter_fr.keys()))
    else:
        print("Determining what to UNK...")
        ordered_counts_en = counter_en.most_common()
        to_unk_en = defaultdict(lambda: True)
        for word, counts in ordered_counts_en[:threshold]:
            to_unk_en[word] = False

        ordered_counts_fr = counter_fr.most_common()
        to_unk_fr = defaultdict(lambda: True)
        for word, counts in ordered_counts_fr[:threshold]:
            to_unk_fr[word] = False
        print("Determined what to UNK.\n")

    # TODO: Get correct maximum sentence length, after setting max.

    def ea_line_mutate(line, word_index_map, index_word_map, index, to_unk, pad_index):
        ##############################################################################################################
        """Function to lowercase all words and strip '\r' and '\n' symbols. Also, implicitly counts word frequency."""
        ##############################################################################################################

        '''Indexes all words in the line (implicitly, in the dataset).'''
        for word in line:
            if (word not in word_index_map):
                if (not to_unk[word]):
                    word_index_map[word] = index.integer
                    index_word_map.append(word)
                index.increment_value()

        line = [word_index_map[word] for word in line]

        return line + [pad_index]*(max_sentence_size - len(line))

    '''Apply line_mutate to all lines in the dataset.'''
    print("Performing last data preprocessing...")
    word_index_map_en = defaultdict(lambda: threshold)
    index_word_map_en = []
    index_en = MutableInt(0)
    to_unk_en = defaultdict(lambda: False) if threshold == 0 else to_unk_en
    data_lines_en = [ea_line_mutate(line, word_index_map_en, index_word_map_en, index_en, to_unk_en, pad_index_en)
                     for line in data_lines_en]

    word_index_map_fr = defaultdict(lambda: threshold)
    index_word_map_fr = []
    index_fr = MutableInt(0)
    to_unk_fr = defaultdict(lambda: False) if threshold == 0 else to_unk_fr
    data_lines_fr = [ea_line_mutate(line, word_index_map_fr, index_word_map_fr, index_fr, to_unk_fr, pad_index_fr)
                     for line in data_lines_fr]
    if (threshold != 0):
        word_index_map_en["UNK"] = threshold
        index_word_map_en.append("UNK")
        word_index_map_fr["UNK"] = threshold
        index_word_map_fr.append("UNK")
    print("Performed last data preprocessing.\n")

    '''Dump data, word_index_map, index_word_map'''

    print("Saving preprocessed data...")

    '''Datalines both languages.'''
    msgpack.dump([data_lines_en, data_lines_fr], open(path_to_data + "/" + training_test + "_" + str(lowercase) +
                                                      '_' + str(max_sentence_size) + '_' + str(threshold) + "_data.both", "wb"), use_bin_type=True)

    # ENGLISH
    '''Word Index Map'''
    msgpack.dump(word_index_map_en, open(path_to_data + "/" + training_test + "_" + str(lowercase) +
                                         '_' + str(max_sentence_size) + '_' + str(threshold) + "_wordIndexMap.en", "wb"), use_bin_type=True)

    '''Index Word Map.'''
    msgpack.dump(index_word_map_en, open(path_to_data + "/" + training_test + "_" + str(lowercase) +
                                         '_' + str(max_sentence_size) + '_' + str(threshold) + "_indexWordMap.en", "wb"), use_bin_type=True)

    # FRENCH
    '''Word Index Map'''
    msgpack.dump(word_index_map_fr, open(path_to_data + "/" + training_test + "_" + str(lowercase) +
                                         '_' + str(max_sentence_size) + '_' + str(threshold) + "_wordIndexMap.fr", "wb"), use_bin_type=True)

    '''Index Word Map.'''
    msgpack.dump(index_word_map_fr, open(path_to_data + "/" + training_test + "_" + str(lowercase) +
                                         '_' + str(max_sentence_size) + '_' + str(threshold) + "_indexWordMap.fr", "wb"), use_bin_type=True)

    print("Saved preprocessed data.\n")


if __name__ == '__main__':
    opt = parse_settings()

    '''path_to_data = osp.join(opt.data_path, opt.dataset, 'training.' + opt.language)

    path_to_data = basic_dataset_preprocess(path_to_data, opt.vocab_size, opt.lowercase)

    preprocess_data_skipgram(path_to_data, opt.window_size, opt.vocab_size + 1, opt.k, opt.save_sequential)'''

    preprocess_data_embedalign(opt.data_path + "/" + opt.dataset + "/", opt.training_test, opt.lowercase,
                               opt.max_sentence_size, opt.vocab_size)
