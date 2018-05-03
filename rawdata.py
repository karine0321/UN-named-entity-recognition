#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

import argparse
import itertools
import json
from multiprocessing import Pool
import nltk
from nltk.chunk import tree2conlltags, conlltags2tree
from nltk.corpus import brown
from nltk.corpus import conll2000
import os
import re
import sys
import time

nltk.download("brown") # for POS tagging
nltk.download("conll2000") # for chunking

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


class posTagger:

    def __init__(self, re_expressions):
        self.re_expressions = re_expressions # regex patterns for the RegexpTagger

    def tagger(self, sentence_obj):
        """
        This method follows the NLTK Book, Chapter 5: http://www.nltk.org/book/ch05.html
        Bigram, unigram, and regex taggers are used before the default tagger assigns
        the most frequent tag from the Brown corpus
        Args:
            sentence_obj : Sentence object
        Returns:
            list of (word, tag) tuples
        """
        trained_sentences = brown.tagged_sents()

        defaultTagger = nltk.DefaultTagger(self.get_default_tag(trained_sentences))
        reTagger = nltk.RegexpTagger(self.re_expressions, backoff = defaultTagger)
        uniTagger = nltk.UnigramTagger(trained_sentences, backoff = reTagger)
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


class NEChunker():
    """
    Uses the built-in NLTK NE tagger (will generate a baseline of NEtags)
    """
    def __init__(self, posTaggedSentence):
        tree = nltk.ne_chunk(posTaggedSentence.pos)
        iob_tags = tree2conlltags(tree)

        # insert code to change IOB tag format

        self.chunk_tags = iob_tags


class BigramChunker(nltk.ChunkParserI):
    """
    Label sentence with chunk tags based on POS tags
    Methods taken from the NLTK book, Chapter 7: https://www.nltk.org/book/ch07.html
    """

    def __init__(self, train_sents):
        self.train_sents = train_sents

        self.unitagger = nltk.UnigramTagger(self.train_sents)

    @classmethod
    def read_from_json(cls, json_chunked_data_fn):
        """
        Load train_sents from a JSON file of manually tagged chunk data
        """

        with open(json_chunked_data_fn, "r") as f:
            data = json.load(f)

        # convert inner list to tuples of (W,POS,CHUNK) and keep only POS,CHUNK
        all_tags = [[tuple(w) for w in s] for s in data]
        train_sents = [[(p,c) for w,p,c in s] for s in all_tags]

        return train_sents

    @classmethod
    def load_nltk_chunked_sentences(cls):
        """
        Load CONLL2000 chunktagged sentences and convert from nltk.tree format
        Returns:
            List of lists where inner list is [(pos, chunk_tag), ... ] representing one sentence
        """
        train_sents = [
            [(pos_tag, chunk_tag) for word, pos_tag, chunk_tag in nltk.chunk.tree2conlltags(sentence)]
            for sentence in conll2000.chunked_sents()
                ]

        return train_sents


    def tag_sentences(self, posTaggedSentence):
        """
        Chunktag a posTaggedSentence object
        """
        pos_tags = [pos for (word, pos) in posTaggedSentence.pos]

        pos_and_chunk_tags = self.unitagger.tag(pos_tags)

        # # Use this block to output word,pos,chunk tags
        # chunk_tags = [chunk for (pos, chunk) in pos_and_chunk_tags]
        # words = [word for (word, pos) in posTaggedSentence.pos]
        # all_tags = list(zip(words, pos_tags, chunk_tags))
        # return all_tags

        return pos_and_chunk_tags


class chunkTaggedSentence(Sentence):
    """
    Like a posTaggedSentence, but with chunk tags
    """

    def __init__(self, posTaggedSentence, pos_and_chunk_tags):

        super().__init__(posTaggedSentence.words_objects, posTaggedSentence.list_of_words)
        self.pos = posTaggedSentence.pos
        self.chunk_tags = pos_and_chunk_tags
