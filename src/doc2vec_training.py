import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import string
import re
import time
from datetime import datetime
from collections import Counter

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.metrics import CategoricalAccuracy, CosineSimilarity, F1Score

from joblib import Parallel, delayed

#from method_main import *
#from method_json import *

import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

'''
corpus_file = '/Users/sro/2024ML-Research/arxiv_classification_project_root/data/corpus.txt'
mapping_file = '/Users/sro/2024ML-Research/arxiv_classification_project_root/data/mapping.txt'

# Ensure paths exist
assert Path(corpus_file).is_file(), f"Corpus file does not exist: {corpus_file}"
assert Path(mapping_file).is_file(), f"Mapping file does not exist: {mapping_file}"

# Function to yield TaggedDocument objects from corpus and mapping files
def read_corpus(corpus_file, mapping_file):
    """Function to yield TaggedDocument objects from corpus and mapping files."""
    try:
        with open(corpus_file, 'r') as corp_file, open(mapping_file, 'r') as map_file:
            for map_line, corp_line in zip(map_file, corp_file):
                doc_id = int(map_line.strip().split(',')[0])  # Adjust delimiter to extract just the ID
                yield TaggedDocument(words=corp_line.split(), tags=[doc_id])
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)

# Store the output of read_corpus in a list
tagged_documents = list(read_corpus(corpus_file, mapping_file))

VECTOR_SIZE = 400
WINDOW_SIZE = 10
MIN_COUNT = 3
alpha = 0.02
min_alpha = 0.0001
cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {cores}")

start_time = time.perf_counter()

d2v_model = Doc2Vec(vector_size=VECTOR_SIZE, dm = 1, window=WINDOW_SIZE, min_count=MIN_COUNT, sample = 0, workers=cores, alpha=alpha, min_alpha=min_alpha)

#d2v_model = Doc2Vec(vector_size=VECTOR_SIZE, dm = 1, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=WORKERS) # 8 cores seem to be the "optimal" number
# Create TaggedDocuments
d2v_model.build_vocab(tagged_documents)
d2v_model.train(tagged_documents, total_examples=d2v_model.corpus_count, epochs = 100)

d2v_model.save("/Users/sro/2024ML-Research/doc2vec_models/doc2vec_model_90%_fulltext_corpus_txt")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
# Get the current timestamp
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# Write the elapsed time and timestamp to the file
with open("/Users/sro/2024ML-Research/doc2vec_models/doc2vec_runtime_summary.txt", 'a') as file:
    file.write(f"Elapsed time: {elapsed_time / 60} minutes \n")
    file.write(f"Timestamp: {current_time}\n")
'''
JSON_8 = "/Users/sro/2024ML-Research/Performance_Metrics/datasets/minor_train_processed.json"
JSON_9 = "/Users/sro/2024ML-Research/Performance_Metrics/datasets/minor_test_processed.json"

df_minor_train = pd.read_json(JSON_8, lines = "True")
df_minor_test = pd.read_json(JSON_9, lines = "True")
formatted_num_rows = "{:,}".format(len(df_minor_train))
print(f"Total number of minor train set rows: {formatted_num_rows}\n")
formatted_num_rows = "{:,}".format(len(df_minor_test))
print(f"Total number of minor test set rows: {formatted_num_rows}\n")

categories = df_minor_train['primary_subfield']
primary_categories = [category.split()[0] for category in categories]
unique_id = [id for id in df_minor_train['paper_id']]

#abstracts = [sentence.split() for sentence in df_minor_train['abstract']]
#titles = [sentence.split() for sentence in df_minor_train['title']]
#fulltexts = [sentence.split() for sentence in df_minor_train['fulltext']]
#data = []
#or fulltexts, title, abstract in zip(fulltexts, titles, abstracts):
#    combined = fulltexts + title + abstract
#    data.append(combined)

data = df_minor_train["processed_title_abstract_fulltext"]

tagged_data_minor = [TaggedDocument(words=doc, tags=[unique_id[i], primary_categories[i]]) for i, doc in enumerate(data)]
tagged_data_dict = [
    {'words': tagged_doc.words, 'tags': tagged_doc.tags}
    for tagged_doc in tagged_data_minor
]
temp_df = pd.DataFrame(tagged_data_dict)
temp_df.to_json(f'Performance_Metrics/datasets/Cornell_minor_dataset_tagged_data.json', orient='records', lines=True)

# Enable logging to monitor training loss
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#tagged_data_minor = pd.read_json('Performance_Metrics/datasets/Cornell_minor_dataset_tagged_data.json', lines = "True")
# Convert the DataFrame back to a list of TaggedDocument objects
##tagged_data_minor_list = [
#   TaggedDocument(words=row['words'], tags=row['tags'])
#   for _, row in tagged_data_minor.iterrows()
#]

VECTOR_SIZE = 300
WINDOW_SIZE = 6
MIN_COUNT = 3
alpha = 0.02
min_alpha = 0.0001
cores = multiprocessing.cpu_count()
print(f"Number of CPU cores: {cores}")

start_time = time.perf_counter()

d2v_model = Doc2Vec(vector_size=VECTOR_SIZE, dm = 1, window=WINDOW_SIZE, min_count=MIN_COUNT, sample = 0, workers=cores, alpha=alpha, min_alpha=min_alpha)

d2v_model.build_vocab(tagged_data_minor)
d2v_model.train(tagged_data_minor, total_examples=d2v_model.corpus_count, epochs = 100)

d2v_model.save("/Users/sro/2024ML-Research/Performance_Metrics/datasets/doc2vec_model_Cornell_minor_100_3_6")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
# Get the current timestamp
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# Write the elapsed time and timestamp to the file
with open("/Users/sro/2024ML-Research/Performance_Metrics/datasets/doc2vec_runtime_summary.txt", 'a') as file:
    file.write(f"Elapsed time: {elapsed_time / 60} minutes \n")
    file.write(f"Timestamp: {current_time}\n")

