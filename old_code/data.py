#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

import argparse
import itertools
from multiprocessing import Pool
import nltk
from nltk.corpus import brown
import os
import re
import sys
import time

nltk.download("brown")

# Helper functions for class Data

def extract_tokens_tags(raw_data):
    """
    Helper function for group_words_sentences. Converts rawdata to List of Lists
    Args:
        raw_data : list of lines, each of which is word\tNEtag
    Returns:
        [ [word, NEtag], [word, NEtag], "\n", ... ], where newlines designate end of sentence
    """
    return [line if line == "\n" else line.rstrip().split("\t") for line in raw_data]

def group_words_sentences(raw_data):
    """
    Returns List of Lists where inner lists are sentences w/ or w/o NEtags. Thanks itertools!
    Args:
        raw_data : list of lines
    Returns:
        token_and_tag_grouped_by_sentence :
            List of lists, where each inner list is a sentence of (word, NEtag) tuples
        tokens_grouped_by_sentence :
            List of lists, where each inner list is a sentence
    """
    grouped_by_sentence = []
    for _, g in itertools.groupby(extract_tokens_tags(raw_data), lambda x: x == "\n"):
        grouped_by_sentence.append(list(g)) # Store group iterator as a list

    token_and_tag_grouped_by_sentence = [g for g in grouped_by_sentence if g != ["\n"]]
    tokens_grouped_by_sentence = [ [i[0] for i in g] for g in grouped_by_sentence if g != ["\n"]]

    return token_and_tag_grouped_by_sentence, tokens_grouped_by_sentence


def convert_data_to_dicts(grouped_by_sentence):
    """
    Args:
        grouped_by_sentence : List of lists, where each inner list is a sentence of (word, NEtag) tuples
    Returns:
        features, tags : lists of dicts.
            Features is list of dicts of word and its features
            Tags is list of dicts of word and its NEtag
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
        self.tokens_grouped_by_sentence = None # [[words from a sentence], ...] input for NLTK POS Tagger
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
        self.NEtags = []
        self.tokens_grouped_by_sentence = []

        for fn in os.listdir(data_dir):
            if fn.endswith(".txt"):
                with open(os.path.join(data_dir, fn), "r") as f:
                    raw_data = f.readlines()

                    token_and_tag_grouped_by_sentence, tokens_grouped_by_sentence = group_words_sentences(raw_data)

                    features, tags = convert_data_to_dicts(token_and_tag_grouped_by_sentence)

                    self.features.extend(features)
                    self.NEtags.extend(tags)
                    self.tokens_grouped_by_sentence.extend(tokens_grouped_by_sentence)


class posTagger:

    def __init__(self, data, re_expressions):
        self.data = data # Load a data object of training or test data
        self.re_expressions = re_expressions # regex patterns for the RegexpTagger

    def tagger(self, sentence):
        """
        This method follows the NLTK Book, Chapter 5: http://www.nltk.org/book/ch05.html
        Bigram, unigram, and regex taggers are used before the default tagger assigns
        the most frequent tag from the Brown corpus
        Args:
            sentence : list of words
        Returns:
            list of (word, tag) tuples

        """
        trained_sentences = brown.tagged_sents()

        defaultTagger = nltk.DefaultTagger(self.get_default_tag(trained_sentences))
        #reTagger = nltk.RegexpTagger(self.re_expressions, backoff = defaultTagger)
        uniTagger = nltk.UnigramTagger(trained_sentences, backoff = defaultTagger)
        biTagger = nltk.BigramTagger(trained_sentences, backoff = uniTagger)

        return biTagger.tag(sentence)



    def get_default_tag(self, trained_sentences):
        """
        Helper function for tagger. Finds the most frequent tag from the corpus, to use as default
        Args:
            trained_sentences : trained nltk sentences
        Returns:
            str of most frequent tag
        """
        freq_dist = nltk.FreqDist(sent[0][1] for sent in trained_sentences)

        return max(freq_dist)

