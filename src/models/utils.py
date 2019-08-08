#!/usr/bin/env python
# coding: utf-8
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
import subprocess
import logging


def get_sequence_data(settings_dict):

    logger = logging.getLogger(__name__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger.setLevel(logging.INFO)

    PROCESSED_PATH = Path(settings_dict["data_path"] + "processed/")

    SEQUENCE_DATA_FNAME = (
        settings_dict["math_module"] + "-" + settings_dict["train_level"]
    )

    SEQUENCE_DATA_PATH = PROCESSED_PATH / SEQUENCE_DATA_FNAME

    if not SEQUENCE_DATA_PATH.is_file():
        logger.info(
            "Sequence data not found for module << {} >>! Generating sequence data ...".format(
                settings_dict["math_module"]
            )
        )
        subprocess.call(["make sequence_data"], stdout=subprocess.PIPE, shell=True)

    with open("{}.pkl".format(SEQUENCE_DATA_PATH), "rb") as file:
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

    inputs, targets = {}, {}

    inputs["train"] = input_texts_train
    inputs["valid"] = input_texts_valid
    inputs["interpolate"] = input_texts["interpolate"]
    inputs["extrapolate"] = input_texts["extrapolate"]

    targets["train"] = target_texts_train
    targets["valid"] = target_texts_valid
    targets["interpolate"] = target_texts["interpolate"]
    targets["extrapolate"] = target_texts["extrapolate"]

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

    return params, inputs, targets
