#!/usr/bin/env python
# coding: utf-8

import pickle
from sklearn.model_selection import train_test_split


def concatenate_texts(path, pattern):
    file_paths = list(path.glob("{}*.txt".format(pattern)))

    input_texts = []
    target_texts = []

    for file_path in file_paths:
        with open(str(file_path), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")[:-1]

        input_texts.extend(lines[0::2])
        target_texts.extend(["\t" + target_text + "\n" for target_text in lines[1::2]])

    return input_texts, target_texts


def get_sequence_data(settings_dict):
    # define file paths
    sequence_data_path = "data/processed/"

    SEQUENCE_DATA_FILE = (
        settings_dict["math_module"] + "-" + settings_dict["train_level"]
    )

    with open(f"{sequence_data_path}{SEQUENCE_DATA_FILE}.pkl", "rb") as file:
        sequence_data = pickle.load(file)

    # load raw data and split into input (questions) and target (answers)
    input_texts = sequence_data["input_texts"]
    target_texts = sequence_data["target_texts"]

    num_encoder_tokens = sequence_data["num_encoder_tokens"]
    num_decoder_tokens = sequence_data["num_decoder_tokens"]
    max_encoder_seq_length = sequence_data["max_encoder_seq_length"]
    max_decoder_seq_length = sequence_data["max_decoder_seq_length"]

    input_texts_train, input_texts_valid, target_texts_train, target_texts_valid = train_test_split(
        input_texts["train"], target_texts["train"], test_size=0.2, random_state=42
    )

    # Creating a mapping from unique characters to indices
    input_token_index = sequence_data["input_token_index"]
    target_token_index = sequence_data["target_token_index"]

    # parameters for data generators
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

    return (
        params,
        input_texts_train,
        input_texts_valid,
        target_texts_train,
        target_texts_valid,
        input_texts,
        target_texts,
    )
