#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

from data import Data, posTagger

import argparse
import itertools
import nltk
from nltk.corpus import brown
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("training_dir", help = "dit containing training data")
parser.add_argument("test_dir", help = "dir containing tagged test data")
args = parser.parse_args()

#training = Data(args.training_dir, True)

tagger = posTagger()

print(tagger.tagged_corpus)