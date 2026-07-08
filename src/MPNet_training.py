import os
import sys
import logging
import multiprocessing
import numpy as np
import pandas as pd
import json
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from itertools import islice
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Enable logging to monitor training loss
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

start_time = time.perf_counter()


JSON_8 = "/Users/sro/2024ML-Research/Performance_Metrics/datasets/minor_train_processed.json"
JSON_9 = "/Users/sro/2024ML-Research/Performance_Metrics/datasets/minor_test_processed.json"
df_minor_train = pd.read_json(JSON_8, lines = "True")
df_minor_test = pd.read_json(JSON_9, lines = "True")
formatted_num_rows = "{:,}".format(len(df_minor_train))
print(f"Total number of minor train set rows: {formatted_num_rows}\n")
formatted_num_rows = "{:,}".format(len(df_minor_test))
print(f"Total number of minor test set rows: {formatted_num_rows}\n")

# If splitting the DataFrame into training and test sets, reset the indices to ensure they are continuous and match correctly.
df_minor_train = df_minor_train.reset_index(drop=True)

categories = df_minor_train['primary_subfield']
primary_categories_train = [category.split()[0] for category in categories]
fulltexts_train = df_minor_train['fulltext']

titles_and_abstracts_train = []
abstracts = [sentence for sentence in df_minor_train['abstract']]
titles = [sentence for sentence in df_minor_train['title']]
for title, abstract in zip(titles, abstracts):
    combined = title + abstract
    titles_and_abstracts_train.append(combined)

label_encoder = LabelEncoder()
label_encoder.fit(primary_categories_train)
y_train_integer = label_encoder.transform(primary_categories_train)
num_categories = len(label_encoder.classes_)
#print(label_encoder.classes_)
# y_one_hot = to_categorical(y_integer, num_classes = num_categories)

categories = df_minor_test['primary_subfield']
primary_categories_test = [category.split()[0] for category in categories]
y_test_integer = label_encoder.transform(primary_categories_test)
fulltexts_test = df_minor_test['fulltext']

titles_and_abstracts_test = []
abstracts = [sentence for sentence in df_minor_test['abstract']]
titles = [sentence for sentence in df_minor_test['title']]
for title, abstract in zip(titles, abstracts):
    combined = title + abstract
    titles_and_abstracts_test.append(combined)

X_train_1, X_eval_1, y_train_1, y_eval_1 = train_test_split(fulltexts_train, y_train_integer, test_size=0.2, random_state=42)

train_data = pd.DataFrame({'fulltext': fulltexts_train, 'title_abstract': titles_and_abstracts_train, 'label': y_train_integer})
test_data = pd.DataFrame({'fulltext': fulltexts_test, 'title_abstract': titles_and_abstracts_test, 'label': y_test_integer})

train_data.to_json('/Users/sro/2024ML-Research/MPNet/minor_train.json', orient='records', lines=True)
test_data.to_json('/Users/sro/2024ML-Research/MPNet/minor_test.json', orient='records', lines=True)

print("Train and test data saved as JSON files")


class ArxivDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            label = self.labels[idx]
        except KeyError as e:
            print(f"KeyError: {e}. Check if index {idx} is valid.")
            raise
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

JSON_1 = '/Users/sro/2024ML-Research/MPNet/minor_train.json'
JSON_2 = '/Users/sro/2024ML-Research/MPNet/minor_test.json'
df_minor_train = pd.read_json(JSON_1, lines = "True")
df_minor_test = pd.read_json(JSON_2, lines = "True")
#X_train = df_minor_train['title_abstract'] # use title + abstract for vector inference instead of full text
X_train = df_minor_train['fulltext']
y_train = df_minor_train['label']
num_categories = len(set(y_train))
#X_test = df_minor_test['title_abstract'] # use title + abstract for vector inference instead of full text
X_test = df_minor_test['fulltext']
y_test = df_minor_test['label']

# tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

train_dataset = ArxivDataset(X_train, y_train, tokenizer, max_length=512)
test_dataset = ArxivDataset(X_test, y_test, tokenizer, max_length=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Use DataCollatorWithPadding for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []  # List to store the accuracy of each run

for i in range(5):
    # Initialize the model for each run
    model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-mpnet-base-v2', num_labels=num_categories)
    model.to(device)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f'/Users/sro/2024ML-Research/MPNet/results_run_{i+1}',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'/Users/sro/2024ML-Research/MPNet/logs_run_{i+1}',
        logging_steps=100,  
        eval_strategy="epoch", 
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test dataset
    results_eval = trainer.evaluate(eval_dataset=test_dataset)
    accuracy = results_eval['eval_accuracy']
    results.append(f"Run {i+1}: Test Accuracy = {accuracy:.4f}\n")

# Write results to a text file
with open('/Users/sro/2024ML-Research/MPNet/5-fold_training_results.txt', 'w') as f:
    f.writelines(results)

print("Training complete. Results saved to 'training_results.txt'.")
'''
# model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=num_categories)
model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/all-mpnet-base-v2', num_labels=num_categories)
model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='/Users/sro/2024ML-Research/MPNet/results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/Users/sro/2024ML-Research/MPNet/logs',
    logging_steps=100,  
    eval_strategy="epoch", 
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    #dataloader_num_workers=4
    #fp16=True #fp16 mixed precision requires a GPU (not 'mps')
)
    
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
# Evaluate the model on the test dataset
results = trainer.evaluate(eval_dataset=test_dataset)


model.save_pretrained('/Users/sro/2024ML-Research/MPNet/fine_tuned_mpnet/MPNet_model_fine_tuned_6_epochs_major')
tokenizer.save_pretrained('/Users/sro/2024ML-Research/MPNet/fine_tuned_mpnet/MPNet_tokenizer_6_epochs_major')


#load fine-tuned MPNet
model = AutoModel.from_pretrained('/Users/sro/2024ML-Research/MPNet/fine_tuned_mpnet/MPNet_model_fine_tuned_6_epochs_major')
tokenizer = AutoTokenizer.from_pretrained('/Users/sro/2024ML-Research/MPNet/fine_tuned_mpnet/MPNet_tokenizer_6_epochs_major')

#tokenizer = AutoTokenizer.from_pretrained('/Users/sro/2024ML-Research/MPNet/fine_tuned_scibert/scibert_tokenizer')
#model = AutoModel.from_pretrained('/Users/sro/2024ML-Research/MPNet/fine_tuned_scibert/scibert_model_fine_tuned')

#ensure the model is in evaluation mode
model.eval()

#Clear CUDA cache periodically to free up GPU memory.
torch.cuda.empty_cache()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use DataParallel for multi-GPU processing
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for processing")
    model = torch.nn.DataParallel(model)

model.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)

def batched(iterable, n):
    it = iter(iterable)
    while (batch := list(islice(it, n))):
        yield batch
        
# Function to compute embeddings
def compute_embeddings(texts, batch_size=16):
    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
            model_output = model(**encoded_input)
            mean_pooled = mean_pooling(model_output, encoded_input['attention_mask'])
            normalized_embeddings = F.normalize(mean_pooled, p=2, dim=1)
            embeddings.append(normalized_embeddings.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

# Compute embeddings for the training and evaluation sets
train_embeddings = compute_embeddings(X_train)
test_embeddings = compute_embeddings(X_test)

np.save('/Users/sro/2024ML-Research/MPNet/MPNet_train_embeddings_fulltext_6epochs_major.npy', train_embeddings)
np.save('/Users/sro/2024ML-Research/MPNet/MPNet_test_embeddings_fulltext_6epochs_major.npy', test_embeddings)

print("Embeddings computation completed.")
'''

end_time = time.perf_counter()
elapsed_time = end_time - start_time
# Get the current timestamp
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# Write the elapsed time and timestamp to the file
with open("/Users/sro/2024ML-Research/MPNet/results/MPNet_runtime_summary.txt", 'a') as file:
    file.write(f"Elapsed time: {elapsed_time / 60} minutes \n")
    file.write(f"Timestamp: {current_time}\n")