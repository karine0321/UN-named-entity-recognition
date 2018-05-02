#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

from data import Data, posTagger

import argparse
import itertools
import json
from multiprocessing import Pool
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("training_dir", help = "dit containing training data")
parser.add_argument("test_dir", help = "dir containing tagged test data")
parser.add_argument("features_dir", help = "dir to write features data to")
args = parser.parse_args()

if not os.path.exists(args.features_dir):
    os.makedirs(args.features_dir)

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


if __name__ == '__main__':

    pool = Pool(os.cpu_count() - 1)

    training = Data(args.training_dir, True)

    tagger = posTagger(training, re_expressions)

    # print(len(tagger.data.tokens_grouped_by_sentence))

    # for sentence in tagger.data.tokens_grouped_by_sentence:
    #     print(tagger.tagger(sentence))

    # results = pool.map(tagger.tagger, tagger.data.tokens_grouped_by_sentence)

    for index, result in enumerate(pool.imap(tagger.tagger, tagger.data.tokens_grouped_by_sentence)):
        tagger.data.features[index]["pos_tag"] = result[1]
        with open(os.path.join(args.features_dir, f'{index}.json'), 'w') as f:
            json.dump(tagger.data.features[index], f)


    with open("tagger_results.json", "w") as f:
        json.dump(tagger.data.features, f)

    with open("NEtags.json", "w") as f:
        json.dump(tagger.data.NEtags, f)
