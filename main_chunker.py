#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 2: Chunking
# This script imports posTaggedSentence objects,
# chunk tags them,
# and outputs chunkTaggedSentence objects

from sentences import RawDocument, Sentence, Token, posTaggedSentence, chunkTaggedSentence
from taggers import NEChunker, UnigramChunker, posTagger

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
parser.add_argument("posTagged_JSON_dir", help = "dir to read posTagged Sentence object JSON files from")
parser.add_argument("manually_chunked_JSON", help = "dir of JSONs of manually tagged chunk tags")
args = parser.parse_args()


if __name__ == '__main__':

    # import pos tagged data and convert to list of posTaggedSentence objects
    imported_pos_tagged_sentences = []

    for fn in os.listdir(args.posTagged_JSON_dir):
            imported_pos_tagged_sentences.append(
                posTaggedSentence(os.path.join(args.posTagged_JSON_dir, fn))
            )

    # Initialize chunker

    # chunker = UnigramChunker(UnigramChunker.read_from_json(args.manually_chunked_JSON))

    chunker = UnigramChunker(UnigramChunker.load_nltk_chunked_sentences())

    for sentence in imported_pos_tagged_sentences:
        chunk_tags = chunker.tag_sentences(sentence)

        s = chunkTaggedSentence(sentence, chunk_tags)

        print(s.list_of_words)

    # # Use this block to output formatted word,pos,chunk data for manual chunk tagging
    # out = []
    # for sentence in imported_sentences:
    #     out.append(chunker.tagger(sentence))
    # with open("chunks_manual.json", "w") as f:
    #     json.dump(out, f)