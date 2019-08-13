#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import pickle
import json
import matplotlib.pyplot as plt
from lstm import LSTM_Simple
from metrics import exact_match_metric
from callbacks import NValidationSetsCallback, GradientLogger
from generator import DataGenerator, DataGeneratorSeq

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

print(tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())

def concatenate_texts(path, pattern):
    file_paths = list(path.glob('{}*.txt'.format(pattern)))

    input_texts = []
    target_texts = []

    for file_path in file_paths:
        with open(str(file_path), 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]

        input_texts.extend(lines[0::2])
        target_texts.extend(['\t' + target_text + '\n' for target_text in lines[1::2]])

    return input_texts, target_texts

# Load settings
settings_path = Path('settings/settings.json')

with open(str(settings_path), 'r') as file:
    settings_dict = json.load(file)
print(settings_dict)

raw_path = Path(settings_dict['data_path'])
interpolate_path = raw_path/'interpolate'
extrapolate_path = raw_path/'extrapolate'
train_easy_path = raw_path/'train-easy/'


# Data settings
math_module = settings_dict["math_module"]
train_level = settings_dict["train_level"]

datasets = {
    'train':(raw_path, 'train-' + train_level + '/' + math_module),
    'interpolate':(interpolate_path, math_module),
    'extrapolate':(extrapolate_path, math_module)
           }

input_texts = {}
target_texts = {}

for k, v in datasets.items():
    input_texts[k], target_texts[k] = concatenate_texts(v[0], v[1])
    print('Length of set {} is {}'.format(k, len(input_texts[k])))

print()
print('INPUT:', input_texts['train'][42])
print('OUTPUT:', target_texts['train'][42].strip())
print()

# Concatenate texts to get text metrics (max length, number of unique tokens, etc.):

all_input_texts = sum(input_texts.values(), [])
all_target_texts = sum(target_texts.values(), [])

input_characters = set(''.join(all_input_texts))
target_characters = set(''.join(all_target_texts))

tokens = sorted(list(input_characters | target_characters))
num_tokens = len(tokens)
max_seq_length  = max([len(txt_in) + len(txt_out) for txt_in, txt_out in zip(all_input_texts,all_target_texts)])

print('Number of samples:', len(all_input_texts))
print('number of tokens:', num_tokens)
print('max sequence length:', max_seq_length)

# Delete all texts to realease memory
del all_input_texts
del all_target_texts

# Create train test splits
input_texts_train, input_texts_valid, target_texts_train, target_texts_valid = train_test_split(input_texts['train'], target_texts['train'], test_size=0.2, random_state=42)

print('Number of training samples:', len(input_texts_train))
print('Number of validation samples:', len(input_texts_valid))

# Process text
# Vectorise the text
# Before training, we need to map strings to a numerical representation. Create two lookup tables: one mapping question characters to numbers, and another for answer characters to number.
# Creating a mapping from unique characters to indices
token_index = dict([(char, i) for i, char in enumerate(tokens)])

print(token_index)


# Create keras data generator
# Parameters
params = {'batch_size': settings_dict["batch_size"],
          'max_seq_length': max_seq_length,
          'num_tokens': num_tokens,
          'token_index': token_index,
          'num_thinking_steps': settings_dict["thinking_steps"]
         }

training_generator = DataGeneratorSeq(input_texts=input_texts_train, target_texts=target_texts_train, **params)
validation_generator = DataGeneratorSeq(input_texts=input_texts_valid, target_texts=target_texts_valid, **params)
interpolate_generator = DataGeneratorSeq(input_texts=input_texts['interpolate'], target_texts=target_texts['interpolate'], **params)
extrapolate_generator = DataGeneratorSeq(input_texts=input_texts['extrapolate'], target_texts=target_texts['extrapolate'], **params)

# Train model

valid_dict = {
    'validation':validation_generator,
    'interpolation': interpolate_generator,
    'extrapolation': extrapolate_generator
}


# Setup callbacks
history = NValidationSetsCallback(valid_dict)
gradient = GradientLogger(live_metrics=['loss', 'exact_match_metric'], live_gaps=10)

epochs = settings_dict['epochs']  # Number of epochs to train for.
latent_dim = settings_dict['latent_dim']  # Latent dimensionality of the encoding space.

if ('saved_model' in settings_dict) and (len(settings_dict['saved_model'])>0):
    model = load_model(settings_dict['saved_model'], compile=False)
else:
    lstm = LSTM_Simple(num_tokens, latent_dim)
    model = lstm.get_model()

adam = Adam(lr=6e-4, beta_1=0.9, beta_2=0.995, epsilon=1e-9, decay=0.0, amsgrad=False, clipnorm=0.1)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[exact_match_metric])

print('start training...')
train_hist = model.fit_generator(training_generator,
                                 epochs=epochs,
                                 callbacks=[history, gradient],
                                 verbose=0,
                                )

plt.plot(train_hist.history['loss'],color='C0', label='train')
plt.plot(train_hist.history['validation_loss'], color='C0', label='valid', linestyle='--')
plt.plot(train_hist.history['extrapolation_loss'], color='C1', label='extra',)
plt.plot(train_hist.history['interpolation_loss'], color='C2', label='inter')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.ylim([0,1])
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(settings_dict['save_path'] + 'losses.png', dpi=300)


plt.plot(train_hist.history['exact_match_metric'],color='C0', label='train')
plt.plot(train_hist.history['validation_exact_match_metric'], color='C0', label='valid', linestyle='--')
plt.plot(train_hist.history['extrapolation_exact_match_metric'], color='C1', label='extra',)
plt.plot(train_hist.history['interpolation_exact_match_metric'], color='C2', label='inter')

plt.xlabel('epochs')
plt.ylabel('exact match metric')
plt.legend(loc='best')
plt.ylim([0,1])
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(settings_dict['save_path'] + 'metrics.png', dpi=300)



with open(settings_dict['save_path']+'experiments_output.pkl','wb') as file:
    pickle.dump(train_hist.history, file)

model.save(settings_dict['save_path']+'model.h5')

with open(settings_dict['save_path']+'settings.json','w') as file:
    json.dump(settings_dict, file)

