#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 3: NER
# This script imports chunkTagged objects,
# adds additional features for NER,
# # and implements an ensemble of Classifiers for NER

from sentences import Sentence, Token, posTaggedSentence, chunkTaggedSentence
from taggers import NEChunker, UnigramChunker, posTagger

import argparse
import itertools
import json
from multiprocessing import Pool
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("chunkTagged_JSON_dir", help = "dir to read chunkTagged Sentence object JSON files from")
parser.add_argument("output_dir", help = "dir to write output to")
args = parser.parse_args()
