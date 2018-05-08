# Named Entity Recognition system for political speech

## A full NLP pipeline

### How to run:

#### Part-of-Speech tagging (on `tagged-test/` and `tagged-training/`) separately:

`python3 main_pos.py input_dir output_dir`

POS tagging using a sequence of NLTK taggers (Bigram, Unigram, Regex, Default)

#####Arguments:

- `input_dir` is a directory containing text files with a token and a Named Entity tag on each line separated by a tab, with a line of whitespace separating sentences
- the script generates `output_dir` containing one JSON file per sentence in the `input_dir`
- Each JSON (file) is a dictionary of (1) a list of the words in the sentence as strings, (2) a list of [word, POS] pairs, and (3) a list of dictionaries, each of which represent a word and several features

#### Chunking

`python3 main_chunker.py input_dir training_chunk_data.json output_dir`

Chunk tagging using an NLTK Unigram Tagger and manually tagged training data

Arguments:

- `input_dir` is the output of the previous step
- `training_chunk_data.json` is manually chunktagged data
- the script generates `output_dir` containing JSON files that correspond to those in `input_dir`, with an additional key-value pair is added to represent a list of [word, pos, chunk tag]

#### Feature preparation
`python3 main_prep_for_ner.py input_dir output_dir`

Adding more features for the classifiers

Arguments:

- `input_dir` is the output of the previous step
- `output_dir` contains JSON files that correspond to those in `input_dir`, with additional features added to the word-dictionaries (see (3) in Step 1)

#### Ensemble Named Entity tagging

`python3 main_ner.py training_dir test_dir num_iterations output_dir predicted_classifications.json true_classifications.json` 

NER with MaxEnt MM, multiclass Naive Bayes, and Decision Tree Classifiers

Arguments:

- `training_dir` is the output of the previous step for training data
- `test_dir` as above, for test data
- `num_iterations` for MaxEnt Classifier
-  `output_dir` contains one output file (technically invalid JSON) generated for each classifier, where each line is a word's dictionary of features and the predicted NER tag


#### Performance
`python3 scorer.py true_classifications.json predicted_classifications.json`

Arguments:
- output of the previous step