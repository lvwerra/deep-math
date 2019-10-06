#!/usr/bin/env python
# coding: utf-8
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path
import subprocess
import logging
import json


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
    num_tokens = sequence_data["num_tokens"]
    max_encoder_seq_length = sequence_data["max_encoder_seq_length"]
    max_decoder_seq_length = sequence_data["max_decoder_seq_length"]
    max_seq_length = sequence_data["max_seq_length"]

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
    token_index = sequence_data["token_index"]

    # set parameters for data generators
    data_gen_pars = {
        "batch_size": settings_dict["batch_size"],
        "max_encoder_seq_length": max_encoder_seq_length,
        "max_decoder_seq_length": max_decoder_seq_length,
        "max_seq_length": max_seq_length,
        "num_encoder_tokens": num_encoder_tokens,
        "num_decoder_tokens": num_decoder_tokens,
        "num_tokens": num_tokens,
        "input_token_index": input_token_index,
        "target_token_index": target_token_index,
        "token_index": token_index,
        "num_thinking_steps": settings_dict["thinking_steps"],
    }

    return data_gen_pars, input_texts, target_texts


def concatenate_texts(path, pattern):
    """Globs math module text files according to pattern.

    Args:
        path: pathlib Path to text files.
        pattern: pattern to glob on.

    Returns:
        input_texts: list of questions.
        target_texts: list of answers.
    """

    file_paths = list(path.glob("{}*.txt".format(pattern)))

    input_texts = []
    target_texts = []

    for file_path in file_paths:
        with open(str(file_path), "r", encoding="utf-8") as f:
            lines = f.read().split("\n")[:-1]

        input_texts.extend(lines[0::2])
        target_texts.extend(["\t" + target_text + "\n" for target_text in lines[1::2]])

    return input_texts, target_texts


def concatenate_texts_individual(path, pattern):
    file_paths = list(path.glob('{}*.txt'.format(pattern)))
    
    input_texts = {}
    target_texts = {}

    for file_path in file_paths:
        if file_path.stem not in input_texts:
            input_texts[file_path.stem] = []
            target_texts[file_path.stem] = []
        
        with open(str(file_path), 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[:-1]

        input_texts[file_path.stem].extend(lines[0::2])
        target_texts[file_path.stem].extend(['\t' + target_text + '\n' for target_text in lines[1::2]])
        
    return input_texts, target_texts


def get_data(settings_path):
    with open(str(settings_path), 'r') as file:
        settings_dict = json.load(file)

    raw_path = Path(settings_dict['data_path'])/'raw/mathematics_dataset-v1.0/'
    interpolate_path = raw_path/'interpolate'
    extrapolate_path = raw_path/'extrapolate'
    train_easy_path = raw_path/'train-easy/'
    
    
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
        input_texts[k], target_texts[k] = concatenate_texts_individual(v[0], v[1])
    
    return settings_dict, input_texts, target_texts