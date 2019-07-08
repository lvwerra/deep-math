import tensorflow as tf
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        input_texts,
        target_texts,
        max_encoder_seq_length,
        max_decoder_seq_length,
        num_encoder_tokens,
        num_decoder_tokens,
        input_token_index,
        target_token_index,
        num_thinking_steps,
        shuffle=True,
    ):

        self.batch_size = batch_size
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.indexes = list(range(len(self.input_texts)))
        self.num_thinking_steps = num_thinking_steps
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.input_texts) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        return self.__data_generation(indexes)

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.input_texts))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"
        encoder_input_data = np.zeros(
            (self.batch_size, self.max_encoder_seq_length, self.num_encoder_tokens),
            dtype="float32",
        )
        decoder_input_data = np.zeros(
            (
                self.batch_size,
                self.max_decoder_seq_length + self.num_thinking_steps,
                self.num_decoder_tokens,
            ),
            dtype="float32",
        )
        decoder_target_data = np.zeros(
            (
                self.batch_size,
                self.max_decoder_seq_length + self.num_thinking_steps,
                self.num_decoder_tokens,
            ),
            dtype="float32",
        )

        batch_inputs = [self.input_texts[i] for i in indexes]
        batch_targets = [self.target_texts[i] for i in indexes]

        for i, (input_text, target_text) in enumerate(zip(batch_inputs, batch_targets)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[
                    i, t + self.num_thinking_steps, self.target_token_index[char]
                ] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[
                        i,
                        t + self.num_thinking_steps - 1,
                        self.target_token_index[char],
                    ] = 1.0

        return ([encoder_input_data, decoder_input_data], decoder_target_data)
