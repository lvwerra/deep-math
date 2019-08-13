#!/usr/bin/env python
# coding: utf-8
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
import subprocess
import logging


def get_sequence_data(settings_dict):
    """Processes raw input and target texts to extract vocabs, tokens, and sequence lengths.

    Args:
        settings_dict: A dictionary storing configuration settings.

    Returns:
        data_gen_pars: A dictionary of configuration parameters for the Keras data generators.
        input_texts: A dictionary of key:value pairs whose values span train, validation, interpolation, and extrapolation sets.
        target_texts: A dictionary of key:value pairs whose values span train, validation, interpolation, and extrapolation sets.
    """

    logger = logging.getLogger(__name__)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger.setLevel(logging.INFO)

    output_path = Path(settings_dict["data_path"] + "processed/")
    output_file_name = settings_dict["math_module"] + "-" + settings_dict["train_level"]
    output_file_path = (output_path / output_file_name).with_suffix(".pkl")

    if not output_file_path.is_file():
        logger.info(
            "Sequence data not found for module << {} >>! Generating sequence data ...".format(
                settings_dict["math_module"]
            )
        )
        subprocess.call(["make sequence_data"], stdout=subprocess.PIPE, shell=True)

    with open("{}".format(output_file_path), "rb") as file:
        sequence_data = pickle.load(file)

    # load raw data and split into input (questions) and target (answers)
    raw_input_texts = sequence_data["input_texts"]
    raw_target_texts = sequence_data["target_texts"]

    num_encoder_tokens = sequence_data["num_encoder_tokens"]
    num_decoder_tokens = sequence_data["num_decoder_tokens"]
    max_encoder_seq_length = sequence_data["max_encoder_seq_length"]
    max_decoder_seq_length = sequence_data["max_decoder_seq_length"]

    input_texts_train, input_texts_valid, target_texts_train, target_texts_valid = train_test_split(
        raw_input_texts["train"],
        raw_target_texts["train"],
        test_size=0.2,
        random_state=42,
    )

    input_texts, target_texts = {}, {}

    input_texts["train"] = input_texts_train
    input_texts["valid"] = input_texts_valid
    input_texts["interpolate"] = raw_input_texts["interpolate"]
    input_texts["extrapolate"] = raw_input_texts["extrapolate"]

    target_texts["train"] = target_texts_train
    target_texts["valid"] = target_texts_valid
    target_texts["interpolate"] = raw_target_texts["interpolate"]
    target_texts["extrapolate"] = raw_target_texts["extrapolate"]

    # create a mapping from unique characters to indices
    input_token_index = sequence_data["input_token_index"]
    target_token_index = sequence_data["target_token_index"]

    # set parameters for data generators
    data_gen_pars = {
        "batch_size": settings_dict["batch_size"],
        "max_encoder_seq_length": max_encoder_seq_length,
        "max_decoder_seq_length": max_decoder_seq_length,
        "num_encoder_tokens": num_encoder_tokens,
        "num_decoder_tokens": num_decoder_tokens,
        "input_token_index": input_token_index,
        "target_token_index": target_token_index,
        "num_thinking_steps": settings_dict["thinking_steps"],
    }

    return data_gen_pars, input_texts, target_texts
