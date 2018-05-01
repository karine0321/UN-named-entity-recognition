#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

import argparse
import itertools
import nltk
from nltk.corpus import brown
import os
import sys

# Helper functions for class Data

def extract_tokens_tags(raw_data):
    """
    Helper function for group_words_sentences. Converts rawdata to List of Lists
    Args:
        raw_data : list of lines
    Returns:
        [ [word, NEtag], [word, NEtag], "\n", ... ], where newlines designate end of sentence
    """
    return [line if line == "\n" else line.rstrip().split("\t") for line in raw_data]

def group_words_sentences(raw_data):
    """
    Returns List of Lists where inner lists are sentences. Thanks itertools!
    Args:
        raw_data : list of lines
    Returns:
        List of lists, where each inner list contains words from a sentence.
    """

    grouped_by_sentence = []
    for _, g in itertools.groupby(extract_tokens_tags(raw_data), lambda x: x == "\n"):
        grouped_by_sentence.append(list(g)) # Store group iterator as a list

    grouped_by_sentence = [g for g in grouped_by_sentence if g != ["\n"]]

    return grouped_by_sentence

def convert_data_to_dicts(grouped_by_sentence):
    """
    Args:
        grouped_by_sentence : List of lists, where each inner list contains words from a sentence.
    Returns:
        List of dicts, where each dict is a token and its features.
    """
    features = []
    tags = []

    for sentence in grouped_by_sentence:
        for counter, value in enumerate(sentence):

            word, NEtag = value # unpack word and tag
            features.append(
                {
                    "token": word,
                    "sentence_position": counter,
                    "sentence_start": counter == 0,
                    "sentence_end": counter == len(sentence) - 1,
                }
            )
            tags.append(
                {
                    "token": word,
                    "NEtag": NEtag,
                }
            )

    return features, tags


class Data:

    def __init__(self, data_dir, is_training):
        self.is_training = is_training
        self.features = None
        self.sentences = None
        self.NEtags = None
        self.predicted_tags = None # for test data only

        self.load_data(data_dir, self.is_training)

    def load_data(self, data_dir, is_training):
        """
        Args:
            data_dir dir containing .txt files of data
            is_training bool for whether data is training
        """
        self.features = []
        self.tags = []
        for fn in os.listdir(data_dir):
            if fn.endswith(".txt"):
                with open(os.path.join(data_dir, fn), "r") as f:
                    raw_data = f.readlines()

                    tokens_grouped_by_sentence = group_words_sentences(raw_data)

                    features, tags = convert_data_to_dicts(tokens_grouped_by_sentence)

                    self.features.extend(features)
                    self.tags.extend(tags)


class posTagger:

    def __init__(self):
        self.tagged_corpus = None

        self.load_corpus()

    def load_corpus(self):
        self.tagged_corpus = brown.tagged_sents()



# brown_tags = brown.tagged_words(tagset = "universal")
# tag_freq_dist = nltk.FreqDist(tag for (word, tag) in brown_tags)
#print(tag_freq_dist.most_common())

# brown_sentences = brown.tagged_sents()
# bigram_tagger = nltk.BigramTagger(brown_sentences)
# print(bigram_tagger.tag(["Thank", "you", "Mr.", "Secretary", "."]))


