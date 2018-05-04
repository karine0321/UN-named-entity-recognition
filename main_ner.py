#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 4: NER
# This script runs an ensemble of classifiers for NER

from sentences import Sentence, saturatedToken, Token, POSTaggedSentence, ChunkTaggedSentence, saturatedSentence
from taggers import MaxEntNERTagger

import argparse
import itertools
import json
from multiprocessing import Pool
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("training_dir", help = "dir to read preppedforNERSentences JSON files from")
parser.add_argument("test_dir", help = "dir to read preppedforNERSentences training JSON files from")
parser.add_argument("n_iterations", help = "num iters for classifiers")
parser.add_argument("output_dir", help = "dir to write output to")
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)



if __name__ == '__main__':

    # run the classifiers
    me_classifier = MaxEntNERTagger.load_and_format_data(args.training_dir,
        args.test_dir,
        int(args.n_iterations),
        args.output_dir
        )
