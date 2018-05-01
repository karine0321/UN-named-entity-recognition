#!/usr/bin/env python3

# Leslie Huang
# Tag a few things automatically


import argparse
import collections
from geotext import GeoText
import itertools
from nltk.corpus import stopwords
import os
import re
import sys

source_dir = "tagged-training"
new_dir = "tagged-training-pos"
os.makedirs(new_dir)
files = os.listdir(source_dir)


def extract_file(filename):
    with open(os.path.join(source_dir, filename), "r") as f:
        text = f.readlines()
    return text


def split_into_tuples(text):
    text_tag = []

    for line in text:
        text_tag.append(line.split("\t"))

    return text_tag

def add_pos_tags(text_tag):
    result = []

    for element in text_tag:
        if element != ["\n"]:
            element = tag(element)

        result.append(element)
    return result

def tag(element):
    """
    Element: tuple of (word, NERtag)
    """
    word = element[0]
    NERtag = element[1].strip("\n")
    if word in [",", ".", "!", "?", "'", '"', "-", ":", ";", "(", ")", "—"]:
        return (word, ".", NERtag)

    elif word.isdigit():
        return (word, "NUM", NERtag)

    elif word.lower() in ["the", "a", "an", "some", "most", "every", "no", "which"]:
        return (word, "DET", NERtag)

    elif word.lower() in ["he", "their", "our", "her", "it", "they", "its", "my", "I", "us", "we", "his"]:
        return (word, "PRON", NERtag)

    elif word.lower() in ["really", "also", "very", "already", "still", "early", "now"]:
        return (word, "ADV", NERtag)

    elif word.lower() in ["on", "in", "from", "about", "of", "at", "with", "by", "into", "under"]:
        return (word, "ADP", NERtag)

    elif word.lower() in ["and", "that", "because", "or", "but", "if", "while", "although"]:
        return (word, "CONJ", NERtag)

    elif word.lower() in ["global", "new", "special"]:
        return (word, "ADJ", NERtag)

    elif word.lower() in ["is", "was", "can", "be", "are", "am", "would", "could", "will", "wish", "congratulate", "like", "have", "had"]:
        return (word, "VERB", NERtag)

    else:
        return (word, "NOUN", NERtag)

def convert_text_tags(text_tag):
    text = []

    for line in text_tag:
        if line == ["\n"]:
            text.append("\n")
        else:
            text.append("{}\t{}\t{}\n".format(line[0], line[1], line[2]))

    return text


for file in files:
    text = extract_file(file)

    text_tag = split_into_tuples(text)
    text_tag = add_pos_tags(text_tag)

    with open(os.path.join(new_dir, file), "w") as f:
        f.writelines(convert_text_tags(text_tag))