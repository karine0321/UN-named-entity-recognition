#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

import argparse
import itertools
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("training_dir", help = "dit containing training data")
parser.add_argument("test_dir", help = "dir containing tagged test data")
args = parser.parse_args()

def load_data(data_dir, is_training):
"""
Args:
    data_dir dir containing .txt files of data
    is_training bool for whether data is training
"""
    data = []

    for fn in os.listdir(data_dir):
        if fn.endswith(".txt"):
            with open(os.path.join(data_dir, fn), "r") as f:
                raw_data = f.readlines()

                tokens_grouped_by_sentence = group_words_sentences(raw_data)

                print(tokens_grouped_by_sentence)

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
    Group words into sentences (which are lists separated by "\n") from a pos_list. Thanks itertools!
    Args:
        raw_data : list of lines
    Returns:
        List of lists, where each list contains words from a sentence.
    """

    tokens_tags = extract_tokens_tags(raw_data)

    sentences_list = []
    for _, g in itertools.groupby(tokens_tags, lambda x: x == "\n"):
        sentences_list.append(list(g)) # Store group iterator as a list

    sentences_list = [g for g in sentences_list if g != ["\n"]]

    return sentences_list




def untagged_test_data(data):
"""
Create a set of test data without the tags
data
"""
    pass

load_data(args.training_dir, is_training = True)