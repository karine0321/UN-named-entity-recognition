#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

from rawdata import RawDocument, Sentence, Token, posTagger

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


def load(dirname):

    all_documents = []

    # load training texts into a collection of RawDocument objects
    for fn in os.listdir(dirname):
        if fn.endswith(".txt"):
            all_documents.append(
                RawDocument(True, os.path.join(dirname, fn))
            )

    container = []

    # Now convert each sentence into a Sentence of Words
    for document in all_documents:
        for sentence in document.text_grouped_into_sentences:
            words = []
            list_of_words = [i[0] for i in sentence]

            for counter, value in enumerate(sentence):
                    word, NEtag = value

                    words.append(
                        Token(word, counter, counter == 0, counter == len(sentence) - 1, NEtag)
                        )

            container.append(Sentence(words, list_of_words))

    return container


if __name__ == '__main__':


    training_data = load(args.training_dir) # This is a list of Sentences, each Sentence is a list of Tokens

    pool = Pool(os.cpu_count() - 1)


    tagger = posTagger(re_expressions)


    for index, result in enumerate(pool.imap(tagger.tagger, training_data)):
        print(training_data[index].list_of_words)
        print(result)

        # tagger.data.features[index]["pos_tag"] = result[1]
        # with open(os.path.join(args.features_dir, f'{index}.json'), 'w') as f:
        #     json.dump(tagger.data.features[index], f)


    # with open("tagger_results.json", "w") as f:
    #     json.dump(tagger.data.features, f)

    # with open("NEtags.json", "w") as f:
    #     json.dump(tagger.data.NEtags, f)
