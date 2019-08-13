"""
This seq2seq implementation is based on:
https://keras.io/examples/lstm_seq2seq/
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import tensorflow as tf


class LSTM_S2S:
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
