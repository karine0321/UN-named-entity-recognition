#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 3: NER
# This script imports chunkTagged objects,
# adds additional features for NER,
# # and implements an ensemble of Classifiers for NER

from sentences import Sentence, saturatedToken, Token, posTaggedSentence, chunkTaggedSentence, saturatedSentence
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


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)



# Some lists for pattern matching

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

orgs_list = ["United", "Nations", "European", "Union", "NATO", "General", "Assembly", "Security", "Council", "UNICEF"]

progs_list = ["Programme", "Millennium", "Development", "Goals", "Declaration", "Protocol", "Treaty", "Conference", "Agenda", "Mission"]


if __name__ == '__main__':


    # import chunktagged data and convert to list of chunkTaggedSentence objects
    saturated_sentences = []

    for counter, fn in enumerate(os.listdir(args.chunkTagged_JSON_dir)):
        chunked_sentence = chunkTaggedSentence(
                *chunkTaggedSentence.load_from_json(os.path.join(args.chunkTagged_JSON_dir, fn))
            )

        s = saturatedSentence(chunked_sentence, countries_list, orgs_list, progs_list)

        saturated_sentences.append(s)

        # write saturatedSentences to file


        out_dict = {
                "sentence_LOW": s.list_of_words,

                "sentence_words": [{
                    "token": word.token,
                    "sentence_position": word.sentence_position,
                    "sentence_start": word.sentence_start,
                    "sentence_end": word.sentence_end,
                    "NEtag": word.NEtag,
                    "case": word.case,
                    "last_char": word.last_char,
                    "geoText_loc": word.geoText_loc,
                    "known_org": word.known_org,
                    "known_prog": word.known_prog,
                    "hyphen": word.hyphen,
                    "prev_token": word.prev_token,
                    "next_token" = None
                }

                     for word in s.words_objects],

                "chunk_tags": s.chunk_tags,
                "sentence_POS": s.pos,
                }

        with open(os.path.join(args.output_dir, f'{index}.json'), 'w') as f:
            json.dump(out_dict, f)





