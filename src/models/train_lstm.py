#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import glob
import pickle
import pprint
import json
import matplotlib.pyplot as plt
from lstm import LSTM_S2S
from metrics import exact_match_metric
from callbacks import NValidationSetsCallback, GradientLogger
from generator import DataGenerator
from utils import concatenate_texts
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import multiprocessing

print("Using TensorFlow version:", tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())
cpu_count = multiprocessing.cpu_count()
print("Number of CPUs:", cpu_count)

# load settings
settings_path = Path("settings/settings.json")
with open(str(settings_path), "r") as file:
    settings_dict = json.load(file)

print("Settings:")
pprint.pprint(settings_dict)

# define file paths
raw_path = Path(settings_dict["data_path"])
interpolate_path = raw_path / "interpolate"
extrapolate_path = raw_path / "extrapolate"
train_easy_path = raw_path / "train-easy/"

# configure module and train level
math_module = settings_dict["math_module"]
train_level = settings_dict["train_level"]

datasets = {
    "train": (raw_path, "train-" + train_level + "/" + math_module),
    "interpolate": (interpolate_path, math_module),
    "extrapolate": (extrapolate_path, math_module),
}

# load raw data and split into input (questions) and target (answers)
input_texts = {}
target_texts = {}

for k, v in datasets.items():
    input_texts[k], target_texts[k] = concatenate_texts(v[0], v[1])
    print("Length of {} set is {}".format(k, len(input_texts[k])))

print("Sample input:", input_texts["train"][42])
print("Sample output:", target_texts["train"][42].strip())

# flatten
all_input_texts = sum(input_texts.values(), [])
all_target_texts = sum(target_texts.values(), [])

input_characters = set("".join(all_input_texts))
target_characters = set("".join(all_target_texts))

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in all_input_texts])
max_decoder_seq_length = max([len(txt) for txt in all_target_texts])

print("Number of samples:", len(all_input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)


del all_input_texts
del all_target_texts

input_texts_train, input_texts_valid, target_texts_train, target_texts_valid = train_test_split(
    input_texts["train"], target_texts["train"], test_size=0.2, random_state=42
)

print("Number of training samples:", len(input_texts_train))


print("Number of validation samples:", len(input_texts_valid))

# Creating a mapping from unique characters to indices
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# Parameters
params = {
    "batch_size": settings_dict["batch_size"],
    "max_encoder_seq_length": max_encoder_seq_length,
    "max_decoder_seq_length": max_decoder_seq_length,
    "num_encoder_tokens": num_encoder_tokens,
    "num_decoder_tokens": num_decoder_tokens,
    "input_token_index": input_token_index,
    "target_token_index": target_token_index,
    "num_thinking_steps": settings_dict["thinking_steps"],
}


training_generator = DataGenerator(
    input_texts=input_texts_train, target_texts=target_texts_train, **params
)
validation_generator = DataGenerator(
    input_texts=input_texts_valid, target_texts=target_texts_valid, **params
)
interpolate_generator = DataGenerator(
    input_texts=input_texts["interpolate"],
    target_texts=target_texts["interpolate"],
    **params,
)
extrapolate_generator = DataGenerator(
    input_texts=input_texts["extrapolate"],
    target_texts=target_texts["extrapolate"],
    **params,
)


valid_dict = {
    "validation": validation_generator,
    "interpolation": interpolate_generator,
    "extrapolation": extrapolate_generator,
}


history = NValidationSetsCallback(valid_dict)
gradient = GradientLogger(live_metrics=["loss", "exact_match_metric"], live_gaps=10)


epochs = settings_dict["epochs"]  # Number of epochs to train for.
latent_dim = settings_dict["latent_dim"]  # Latent dimensionality of the encoding space.


lstm = LSTM_S2S(num_encoder_tokens, num_decoder_tokens, latent_dim)


model = lstm.get_model()
print(model.summary())


adam = Adam(
    lr=6e-4,
    beta_1=0.9,
    beta_2=0.995,
    epsilon=1e-9,
    decay=0.0,
    amsgrad=False,
    clipnorm=0.1,
)

model.compile(
    optimizer=adam, loss="categorical_crossentropy", metrics=[exact_match_metric]
)

print("Start training...")
train_hist = model.fit_generator(
    training_generator,
    epochs=epochs,
    use_multiprocessing=True,
    workers=cpu_count,
    callbacks=[history, gradient],
    verbose=0,
)

# create and save plot of losses
fig = plt.figure()
plt.plot(train_hist.history["loss"], color="C0", label="train")
plt.plot(
    train_hist.history["validation_loss"], color="C0", label="valid", linestyle="--"
)
plt.plot(train_hist.history["extrapolation_loss"], color="C1", label="extra")
plt.plot(train_hist.history["interpolation_loss"], color="C2", label="inter")

plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="best")
plt.ylim([0, 1])
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig(settings_dict["save_path"] + "losses.png", dpi=300)

# create and save plot of evaluation metrics
fig = plt.figure()
plt.plot(train_hist.history["exact_match_metric"], color="C0", label="train")
plt.plot(
    train_hist.history["validation_exact_match_metric"],
    color="C0",
    label="valid",
    linestyle="--",
)
plt.plot(
    train_hist.history["extrapolation_exact_match_metric"], color="C1", label="extra"
)
plt.plot(
    train_hist.history["interpolation_exact_match_metric"], color="C2", label="inter"
)

plt.xlabel("epochs")
plt.ylabel("exact match metric")
plt.legend(loc="best")
plt.ylim([0, 1])
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig(settings_dict["save_path"] + "metrics.png", dpi=300)

# save callbacks data
with open(settings_dict["save_path"] + "experiments_output.pkl", "wb") as file:
    pickle.dump(train_hist.history, file)

# save model
model.save(settings_dict["save_path"] + "model.h5")

# save settings
with open(settings_dict["save_path"] + "settings.json", "w") as file:
    json.dump(settings_dict, file)

