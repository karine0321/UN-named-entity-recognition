#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

from data import Data, posTagger

import argparse
import itertools
import nltk
from nltk.corpus import brown
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("training_dir", help = "dit containing training data")
parser.add_argument("test_dir", help = "dir containing tagged test data")
args = parser.parse_args()


# These patterns are taken from the NLTK Book, Chapter 5 http://www.nltk.org/book/ch05.html
re_expressions = patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
]

training = Data(args.training_dir, True)

tagger = posTagger(training, re_expressions)

# training = Data(args.training_dir, True, tagger)

print(tagger.data.tokens_grouped_by_sentence)