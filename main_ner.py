#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 4: NER
# This script runs an ensemble of classifiers for NER

from sentences import Sentence, saturatedToken, Token, POSTaggedSentence, ChunkTaggedSentence, saturatedSentence
from taggers import NEChunker, MaxEntNERTagger

import argparse
import itertools
import json
from multiprocessing import Pool
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("preppedNERSentences_dir", help = "dir to read prepped for NER JSON files from")
parser.add_argument("output_dir", help = "dir to write output to")
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if __name__ == '__main__':

    # import chunktagged data and convert to list of SaturatedToken objects
    token_dicts = []

    for fn in os.listdir(args.preppedNERSentences_dir):
        with open(os.path.join(args.preppedNERSentences_dir, fn), "r") as f:
            sentence = json.load(f)

            token_dicts += sentence

    # run the classifiers