#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 5: Scoring
# This script scores the classifiers

from sentences import Sentence, saturatedToken, Token, POSTaggedSentence, ChunkTaggedSentence, saturatedSentence
from taggers import ClassifierData

import argparse
import itertools
import json
from multiprocessing import Pool
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("test_data", help = "formatted test data filename")
parser.add_argument("predictions", help = "predicted classification data filename")
args = parser.parse_args()

with open(args.test_data, "r") as f:
    true_labels = json.load(f)

with open(args.predictions, "r") as f:
    predicted_labels = json.load(f)

print(len(predicted_labels))