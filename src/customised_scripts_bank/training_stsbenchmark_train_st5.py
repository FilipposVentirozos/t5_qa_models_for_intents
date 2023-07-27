"""
This example loads the pre-trained SentenceTransformer model 'nli-distilroberta-base-v2' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
import transformers
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import sys
from pathlib import Path
from os.path import join
import sys
from transformers import HfArgumentParser, training_args
from typing import List, Optional, Tuple


parser = HfArgumentParser((training_args.TrainingArguments))    
training_args = parser.parse_args_into_dataclasses()[0]

p = Path(__file__).parents[4]

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# /print debug information to stdout

# # Check if dataset exsist. If not, download and extract  it
# sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
#
# if not os.path.exists(sts_dataset_path):
#     util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

model_name = 'sentence-transformers/sentence-t5-base'

# model_save_path = join(training_args.output_dir, model_name.split("/")[-1] + '-' + datetime.now().strftime(
    # "%Y-%m-%d_%H-%M-%S"))
model_save_path = join(training_args.output_dir, "st5")

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)
model.max_seq_length = 320
print("Max seq len:", model.max_seq_length)

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
# test_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:

data = join(p, "data", "Auxiliary", "IML", "SentenceComparisons", "sentence_scoring_sents_combined_split.tsv")
with open(data, "r") as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=training_args.per_device_train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training. We skip evaluation in this example
# warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
# logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=int(training_args.num_train_epochs),
          # steps_per_epoch=10,
          optimizer_class=transformers.Adafactor, 
          optimizer_params={'lr': training_args.learning_rate, "relative_step": False},          
          # evaluation_steps=10, # training_args.eval_steps, # 196
          warmup_steps=1_000,
          output_path=model_save_path,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=training_args.eval_steps, # 196, 10
          checkpoint_save_total_limit=1
)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

# model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
# test_evaluator(model, output_path=model_save_path)


