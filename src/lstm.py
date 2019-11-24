import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    Bidirectional,
    Dense,
    Embedding,
    Input,
    TimeDistributed,
    concatenate,
    dot,
)
from tensorflow.keras.models import Model


class SimpleLSTM:
    def __init__(self, num_tokens, latent_dim):
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim

    def get_model(self):
        # Define an input sequence and process it.
        self.lstm_inputs = Input(shape=(None, self.num_tokens))
        if tf.test.is_gpu_available():
            self.lstm = CuDNNLSTM(
                self.latent_dim, return_state=True, return_sequences=True
            )
        else:
            self.lstm = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        lstm_outputs, state_h, state_c = self.lstm(self.lstm_inputs)
        self.lstm_states = [state_h, state_c]
        self.dense = Dense(self.num_tokens, activation="softmax")
        lstm_outputs = self.dense(lstm_outputs)
        self.model = Model(self.lstm_inputs, lstm_outputs)

        return self.model

    def decode_sample(self, input_seq_list, target_token_index, max_sequence_length):

        # reverse the char -> id dictionary for decoding
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items()
        )

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.lstm(
            self.lstm_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = self.dense(decoder_outputs)

        decoder_model = Model(
            [self.lstm_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states,
        )

        # get the results for each sequence in the list
        results = []
        for input_seq in input_seq_list:
            results.append(
                self.decode_sequence(
                    input_seq,
                    decoder_model,
                    target_token_index,
                    reverse_target_char_index,
                    max_sequence_length,
                )
            )
        return results

    def decode_sequence(
        self,
        input_seq,
        decoder_model,
        target_token_index,
        reverse_target_char_index,
        max_sequence_length,
    ):

        # initial state is zero
        states_value = [np.zeros((1, self.latent_dim)), np.zeros((1, self.latent_dim))]

        # feed in the whole input sequence except the last thinking step which output will be used as
        # input for first relevant output_tokens
        _, h, c = decoder_model.predict([input_seq[:, :-1, :]] + states_value)
        states_value = [h, c]
        target_seq = input_seq[:, -1:, :]

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_sequence_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, len(target_token_index)))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]

        return decoded_sentence


class Seq2SeqLSTM:
    def __init__(self, num_encoder_tokens, num_decoder_tokens, latent_dim):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.latent_dim = latent_dim

    def get_model(self):
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        # Use CuDNNLSTM if running on GPU
        if tf.test.is_gpu_available():
            encoder = CuDNNLSTM(self.latent_dim, return_state=True)
        else:
            encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        if tf.test.is_gpu_available():
            self.decoder_lstm = CuDNNLSTM(
                self.latent_dim, return_sequences=True, return_state=True
            )
        else:
            self.decoder_lstm = LSTM(
                self.latent_dim, return_sequences=True, return_state=True
            )
        decoder_outputs, _, _ = self.decoder_lstm(
            self.decoder_inputs, initial_state=self.encoder_states
        )
        self.decoder_dense = Dense(self.num_decoder_tokens, activation="softmax")
        decoder_outputs = self.decoder_dense(decoder_outputs)

        return Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)


class AttentionLSTM:
    def __init__(
        self,
        num_encoder_tokens,
        num_decoder_tokens,
        max_encoder_seq_length,
        max_decoder_seq_length,
        num_encoder_units,
        num_decoder_units,
        embedding_dim,
    ):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.num_encoder_units = num_encoder_units
        self.num_decoder_units = num_decoder_units
        self.embedding_dim = embedding_dim

    def get_model(self):
        # encoder_inputs shape == (batch_size, encoder_seq_length)
        self.encoder_inputs = Input(shape=(None,))
        # encoder_emb shape == (batch_size, encoder_seq_length, embedding_dim)
        encoder_emb = Embedding(
            self.num_encoder_tokens + 1, self.embedding_dim, mask_zero=True
        )(self.encoder_inputs)
        # encoder shape == (batch_size, encoder_seq_length, num_encoder_units)
        self.encoder_outputs = Bidirectional(
            LSTM(self.num_encoder_units, return_sequences=True, unroll=False)
        )(encoder_emb)
        self.encoder_outputs = Dense(self.num_decoder_units)(self.encoder_outputs)
        # encoder_last shape == (batch_size, num_decoder_units)
        self.encoder_last = self.encoder_outputs[:, -1, :]
        self.encoder_last.set_shape([None, self.num_decoder_units])

        # decoder_inputs shape == (batch_size, decoder_seq_length)
        self.decoder_inputs = Input(shape=(None,))
        # decoder_emb shape == (batch_size, decoder_seq_length, embedding_dim)
        decoder_emb = Embedding(
            self.num_decoder_tokens + 1, self.embedding_dim, mask_zero=True
        )(self.decoder_inputs)
        # decoder_outputs shape == (batch_size, decoder_seq_length, num_decoder_units)
        decoder_outputs = LSTM(
            self.num_decoder_units, return_sequences=True, unroll=False
        )(decoder_emb, initial_state=[self.encoder_last, self.encoder_last])

        # attention shape == (batch_size, decoder_seq_length, max_encoder_seq_length)
        attention = dot([decoder_outputs, self.encoder_outputs], axes=[2, 2])
        attention = Activation("softmax", name="attention")(attention)

        # context shape == (batch_size, decoder_seq_length, latent_dim)
        context = dot([attention, self.encoder_outputs], axes=[2, 1])

        # decoder_combined_context shape == (batch_size, decoder_seq_length, latent_dim)
        decoder_combined_context = concatenate([context, decoder_outputs])

        # decoder_outputs shape == (batch_size, decoder_seq_length)
        decoder_outputs = TimeDistributed(
            Dense(self.num_decoder_units, activation="tanh")
        )(decoder_combined_context)
        # decoder_outputs shape == (batch_size, decoder_seq_length, num_decoder_tokens)
        decoder_outputs = TimeDistributed(
            Dense(self.num_decoder_tokens, activation="softmax")
        )(decoder_outputs)

        return Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
