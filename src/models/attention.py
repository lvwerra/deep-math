"""
This attention-based seq2seq model is largely copied from:
https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
"""
import numpy as np
from tensorflow.keras.layers import (
    LSTM,
    Activation,
    Dense,
    Embedding,
    Input,
    TimeDistributed,
    concatenate,
    dot,
)
from tensorflow.keras.models import Model


class LSTMWithAttention:
    def __init__(
        self,
        num_encoder_tokens,
        num_decoder_tokens,
        max_encoder_seq_length,
        max_decoder_seq_length,
        latent_dim,
        embedding_dim,
    ):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

    def get_model(self):
        # encoder_inputs shape == (batch_size, max_encoder_seq_length)
        self.encoder_inputs = Input(shape=(self.max_encoder_seq_length,))
        # encoder_emb shape == (batch_size, max_encoder_seq_length, embedding_dim)
        encoder_emb = Embedding(
            self.num_encoder_tokens + 1,
            self.embedding_dim,
            input_length=self.max_encoder_seq_length,
            mask_zero=True,
        )(self.encoder_inputs)
        # encoder shape == (batch_size, max_encoder_seq_length, latent_dim)
        self.encoder_outputs = LSTM(
            self.latent_dim, return_sequences=True, unroll=True
        )(encoder_emb)
        # encoder_last shape == (batch_size, latent_dim)
        self.encoder_last = self.encoder_outputs[:, -1, :]
        self.encoder_last.set_shape([None, self.latent_dim])

        # decoder_inputs shape == (batch_size, max_decoder_seq_length)
        self.decoder_inputs = Input(shape=(self.max_decoder_seq_length,))
        # decoder_emb shape == (batch_size, max_decoder_seq_length, embedding_dim)
        decoder_emb = Embedding(
            self.num_decoder_tokens + 1,
            self.embedding_dim,
            input_length=self.max_decoder_seq_length,
            mask_zero=True,
        )(self.decoder_inputs)
        # decoder_outputs shape == (batch_size, max_decoder_seq_length, latent_dim)
        decoder_outputs = LSTM(self.latent_dim, return_sequences=True, unroll=True)(
            decoder_emb, initial_state=[self.encoder_last, self.encoder_last]
        )

        # attention shape == (batch_size, max_decoder_seq_length, max_encoder_seq_length)
        attention = dot([decoder_outputs, self.encoder_outputs], axes=[2, 2])
        attention = Activation("softmax", name="attention")(attention)

        # context shape == (batch_size, max_decoder_seq_length, latent_dim)
        context = dot([attention, self.encoder_outputs], axes=[2, 1])

        # decoder_combined_context shape == (batch_size, max_decoder_seq_length, 2 * latent_dim)
        decoder_combined_context = concatenate([context, decoder_outputs])

        # decoder_outputs shape == (batch_size, max_decoder_seq_length)
        decoder_outputs = TimeDistributed(Dense(self.latent_dim, activation="tanh"))(
            decoder_combined_context
        )
        # decoder_outputs shape == (batch_size, max_decoder_seq_length, num_decoder_tokens)
        decoder_outputs = TimeDistributed(
            Dense(self.num_decoder_tokens, activation="softmax")
        )(decoder_outputs)

        return Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)

    def decode_sample(self, input_seq_list, target_token_index, max_sequence_length):
        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)

        decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states,
        )

        # reverse the char -> id dictionary for decoding
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items()
        )

        # get the results for each sequence in the list
        results = []
        for input_seq in input_seq_list:
            results.append(
                self.decode_sequence(
                    input_seq,
                    encoder_model,
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
        encoder_model,
        decoder_model,
        target_token_index,
        reverse_target_char_index,
        max_sequence_length,
    ):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index["\t"]] = 1.0

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
