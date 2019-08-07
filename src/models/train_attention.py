#!/usr/bin/env python
# coding: utf-8

import json
import logging
import absl.logging
import multiprocessing
import os
import pickle
from pathlib import Path

import click
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from attention import LSTMWithAttention
from callbacks import GradientLogger, NValidationSetsCallback
from generator import DataGeneratorAttention
from metrics import exact_match_metric_index

from utils import get_sequence_data


@click.command()
@click.option("--settings", default="settings_local.json")
def main(settings):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # load settings
    settings_path = Path("settings/" + settings)
    with open(str(settings_path), "r") as file:
        settings_dict = json.load(file)

    logger.info(
        f"Training attention-based model on math module: {settings_dict['math_module']} and difficulty level: {settings_dict['train_level']}"
    )

    logger.info(f"Using TensorFlow version: {tf.__version__}")
    logger.info(f"GPU Available: {tf.test.is_gpu_available()}")
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"Number of CPUs: {cpu_count}")

    params, input_texts_train, input_texts_valid, target_texts_train, target_texts_valid, input_texts, target_texts = get_sequence_data(
        settings_dict
    )

    training_generator = DataGeneratorAttention(
        input_texts=input_texts_train, target_texts=target_texts_train, **params
    )
    validation_generator = DataGeneratorAttention(
        input_texts=input_texts_valid, target_texts=target_texts_valid, **params
    )
    interpolate_generator = DataGeneratorAttention(
        input_texts=input_texts["interpolate"],
        target_texts=target_texts["interpolate"],
        **params,
    )
    extrapolate_generator = DataGeneratorAttention(
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
    gradient = GradientLogger(
        live_metrics=["loss", "exact_match_metric_index"], live_gaps=10
    )

    epochs = settings_dict["epochs"]  # Number of epochs to train for.
    latent_dim = settings_dict[
        "latent_dim"
    ]  # Latent dimensionality of the encoding space.
    embedding_dim = settings_dict[
        "embedding_dim"
    ]  # embedding dimensionality of the encoding space.

    lstm = LSTMWithAttention(
        params["num_encoder_tokens"],
        params["num_decoder_tokens"],
        params["max_encoder_seq_length"],
        params["max_decoder_seq_length"],
        latent_dim,
        embedding_dim,
    )

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
        optimizer=adam,
        loss="categorical_crossentropy",
        metrics=[exact_match_metric_index],
    )

    # directory where the checkpoints will be saved
    checkpoint_dir = settings_dict["save_path"] + "training_checkpoints"
    # name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    print("Start training...")
    # workers = cpu_count // 2 and no multiprocessing?
    train_hist = model.fit_generator(
        training_generator,
        epochs=epochs,
        use_multiprocessing=False,
        workers=cpu_count // 2,
        callbacks=[history, gradient, checkpoint_callback],
        verbose=0,
    )

    # create and save plot of losses
    plt.figure()
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
    plt.figure()
    plt.plot(train_hist.history["exact_match_metric_index"], color="C0", label="train")
    plt.plot(
        train_hist.history["validation_exact_match_metric_index"],
        color="C0",
        label="valid",
        linestyle="--",
    )
    plt.plot(
        train_hist.history["extrapolation_exact_match_metric_index"],
        color="C1",
        label="extra",
    )
    plt.plot(
        train_hist.history["interpolation_exact_match_metric_index"],
        color="C2",
        label="inter",
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


if __name__ == "__main__":
    log_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # workaround for abseil issue https://github.com/tensorflow/tensorflow/issues/26691
    absl.logging.get_absl_handler().setFormatter(log_fmt)

    main()
