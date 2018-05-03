#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 2: Chunking

from rawdata import Sentence, Token, posTagger, posTaggedSentence, NEChunker, BigramChunker

import argparse
import itertools
import json
from multiprocessing import Pool
import nltk
from nltk.corpus import conll2000
import os
import re
import sys

nltk.download("conll2000") # for chunking

parser = argparse.ArgumentParser()
parser.add_argument("features_dir", help = "dir to write features data to")
args = parser.parse_args()


if __name__ == '__main__':

    # import pos tagged data and convert to list of posTaggedSentence objects
    imported_sentences = []

    for fn in os.listdir(args.features_dir):
            imported_sentences.append(
                posTaggedSentence(os.path.join(args.features_dir, fn))
            )

    chunker = BigramChunker(conll2000.chunked_sents(chunk_types=['NP']))



    out = []
    for sentence in imported_sentences:
        out.append(chunker.bigram_chunk_tagger(sentence))

    with open("chunks_manual.json", "w") as f:
        json.dump(out, f)