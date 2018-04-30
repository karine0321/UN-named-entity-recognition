#!/usr/bin/env python3

# Leslie Huang
# Tag a few things automatically


# O - none
# I-LOC - location
# I-PER - person
# I-ORG - organization
# I-MISC - misc named entity

import argparse
import collections
from geotext import GeoText
import itertools
from nltk.corpus import stopwords
import os
import re
import sys

directory = "test"
files = os.listdir(directory)

countries_list = ["Afghanistan", "Algeria", "Argentina", "Austalia",
    "Bahrain", "Bhutan", "Brazil", "Bulgaria",
    "Canada", "Chile", "Chad", "China", "Colombia",
    "Denmark", "Egypt", "El Salvador", "Ethiopia", "France",
    "Germany", "Ghana", "Iceland", "India", "Iraq", "Japan",
    "Kuwait", "Libya", "Mali", "Mexico", "Myanmar", "Morocco", "Mongolia",
    "New Zealand", "Norway", "Pakistan", "Peru", "Sierra Leone",
    "Russian Federation", "Switzerland", "Sweden", "Turkey",
    "United States of America", "United Kingdom of Great Britain and Northern Ireland",
    "Zimbabwe"]

orgs_list = ["United", "Nations", "European", "Union", "NATO", "General", "Assembly", "Security", "Council"]

progs_list = ["Programme", "Millennium", "Development", "Goals", "Declaration"]

def extract_file(filename):
    with open(os.path.join(directory, filename), "r") as f:
        text = f.readlines()

    return text

def convert_tuples(text):
    text_tag = []

    for line in text:
        if line != "\n":
            text_tag.append((line.strip(), "O"))
        else:
            text_tag.append(("\n", None))

    return text_tag

def change_tags(text_tag):
    new_tags = []
    for (word, tag) in text_tag:
        tag = word_country_tag(word, tag)
        tag = tag_stopwords(word, tag)
        tag = tag_orgs(word, tag)
        tag = tag_progs(word, tag)
        tag = tag_geo(word, tag)

        new_tags.append((word, tag))

    return new_tags

def word_country_tag(word, tag):
    if word in " ".join(countries_list).split(" "):
        return "I-LOC"
    else:
        return tag

def tag_stopwords(word, tag):
    if word.lower() in stopwords.words("english"):
        return "O"
    else:
        return tag

def tag_orgs(word, tag):
    if word in orgs_list:
        return "I-ORG"
    else:
        return tag

def tag_progs(word, tag):
    if word in progs_list:
        return "I-MISC"
    else:
        return tag

def tag_geo(word, tag):
    if GeoText(word).cities or GeoText(word).countries:
        return "I-LOC"
    else:
        return tag

def convert_text_tags(text_tag):
    text = []

    for (word, tag) in text_tag:
        if tag is not None:
            text.append("{}\t{}\n".format(word, tag))
        else:
            text.append("\n")

    return text


for file in files:
    text = extract_file(file)

    text_tag = convert_tuples(text)

    text = convert_text_tags(change_tags(text_tag))

    with open(file, "w") as f:
        f.writelines(text)