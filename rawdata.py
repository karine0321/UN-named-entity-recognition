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


def split_lines_into_token_tag(raw_text):
    """
    Helper function for group_text_into_sentences. Converts rawdata to List of Lists
    Args:
        raw_text : list of lines, each of which is word\tNEtag
    Returns:
        [ [word, NEtag], [word, NEtag], "\n", ... ], where newlines designate end of sentence
    """
    return [line if line == "\n" else line.rstrip().split("\t") for line in raw_text]

def group_text_into_sentences(raw_text):
    """
    Returns List of Lists where inner lists are sentences w/ or w/o NEtags. Thanks itertools!
    Args:
        raw_data : list of lines
    Returns:
        List of lists, where each inner list is a sentence of (word, NEtag) tuples
    """
    grouped_by_sentence = []

    for _, g in itertools.groupby(split_lines_into_token_tag(raw_text), lambda x: x == "\n"):
        grouped_by_sentence.append(list(g)) # Store group iterator as a list

    return [g for g in grouped_by_sentence if g != ["\n"]]


class RawDocument():
    """
    Initial container for a single (training or test) text.
    """
    def __init__(self, is_training, filepath):
        self.is_training = is_training
        self.raw_text = self.load(filepath) # Is a list of lines from the text
        self.text_grouped_into_sentences = group_text_into_sentences(self.raw_text)

    def load(self, filepath):
        with open(filepath, "r") as f:
            raw_text = f.readlines()

        return raw_text


class Sentence():
    """
    Sentence is a container for a list of Word objects
    """
    def __init__(self, words_objects, list_of_words = None):
        self.words_objects = words_objects # List of Word objects
        self.list_of_words = list_of_words # List of words in the sentence

class Token():
    """
    Container for a single token and its features
    """
    def __init__(self, token, sentence_position, sentence_start, sentence_end, NEtag):
        self.token = token
        self.sentence_position = sentence_position
        self.sentence_start = sentence_start
        self.sentence_end = sentence_end
        self.NEtag = NEtag



class posTagger:

    def __init__(self, re_expressions):
        self.re_expressions = re_expressions # regex patterns for the RegexpTagger

    def tagger(self, sentence_obj):
        """
        This method follows the NLTK Book, Chapter 5: http://www.nltk.org/book/ch05.html
        Bigram, unigram, and regex taggers are used before the default tagger assigns
        the most frequent tag from the Brown corpus
        Args:
            sentence_obj : Sentence
        Returns:
            list of (word, tag) tuples

        """
        trained_sentences = brown.tagged_sents()

        defaultTagger = nltk.DefaultTagger(self.get_default_tag(trained_sentences))
        #reTagger = nltk.RegexpTagger(self.re_expressions, backoff = defaultTagger)
        uniTagger = nltk.UnigramTagger(trained_sentences, backoff = defaultTagger)
        biTagger = nltk.BigramTagger(trained_sentences, backoff = uniTagger)

        return biTagger.tag(sentence_obj.list_of_words)

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

