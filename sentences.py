#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

import argparse
import itertools
import json
from multiprocessing import Pool
import nltk
import os
import re
import sys
import time

# Helper functions for RawDocument class

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


class Sentence():
    """
    Sentence is a container for a list of Word objects
    """
    def __init__(self, words_objects, list_of_words):
        self.words_objects = words_objects # List of Word objects
        self.list_of_words = list_of_words # List of words in the sentence


class posTaggedSentence(Sentence):
    """
    Like a Sentence, but with an attribute for POS tags.
    """
    def __init__(self, sentence_json_filepath):
        words_objects, list_of_words, sentence_POS = self.load(sentence_json_filepath)

        super().__init__(words_objects, list_of_words)
        self.pos = sentence_POS

    def load(self, sentence_json_filepath):
        """
        Loads each Sentence JSON that was written to file by posTagger
        """
        sentence_json = json.load(open(sentence_json_filepath))

        words_objects = [Token(w["token"], w["sentence_position"], w["sentence_start"], w["sentence_end"], w["NEtag"]) for w in sentence_json["sentence_words"]]

        return words_objects, sentence_json["sentence_LOW"], sentence_json["sentence_POS"]


class chunkTaggedSentence(Sentence):
    """
    Like a posTaggedSentence, but with chunk tags
    """

    def __init__(self, posTaggedSentence, pos_and_chunk_tags):

        super().__init__(posTaggedSentence.words_objects, posTaggedSentence.list_of_words)
        self.pos = posTaggedSentence.pos
        self.chunk_tags = pos_and_chunk_tags
