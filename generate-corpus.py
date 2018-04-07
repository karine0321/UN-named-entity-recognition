#!/usr/bin/env python3

# Leslie Huang

import argparse
import collections
import itertools
import numpy as np
import os
import re
import sys
import pandas as pd


un_corpus = pd.read_csv("un_corpus.csv")
un_corpus.drop(["filename", "record_name", "speaker_notes"], axis = 1)

# So that training/test data cover some overlapping countries/years, use a subset of the original data
# countries_list = np.random.choice(un_corpus["country"].unique(), 50, False)
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

year_over_2000 = un_corpus["year"] >= 2000
countries_subset = un_corpus["country"].isin(countries_list)
un_corpus_2000 = un_corpus[year_over_2000 & countries_subset]

# Select a sample of 75 documents (50 for training, 25 for test)

sample_corpus = un_corpus_2000.sample(n = 75, replace = False, random_state = 1)

training_corpus = sample_corpus.head(50)

test_corpus = sample_corpus.tail(25)

# Process each text (row)

def prepare_for_tagging(df, directory):
    for row in df.itertuples():
        text = row.text.split()

        # one word per line, including punctuation
        split_by_lines = re.sub(r"([,;'])", r"\n\1", "\n".join(text))
        split_by_lines = re.sub(r"([.?!])", r"\n\1\n", split_by_lines) # extra newline following EOS punctuation
        split_by_lines = re.sub(r"('s)", r"\n\1", split_by_lines) # possessive 's on separate line
        split_by_lines = re.sub(r"(n't)", r"\n\1", split_by_lines) # n't contraction on separate line
        split_by_lines = re.sub(r"Mr\n.\n\n", r"Mr.\n", split_by_lines) # make Mr. one word

        filename = "{}_{}_{}.txt".format(row.ID, row.year, row.country)

        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, filename), "w") as f:
            f.write(split_by_lines)

prepare_for_tagging(training_corpus, "training")
prepare_for_tagging(test_corpus, "test")