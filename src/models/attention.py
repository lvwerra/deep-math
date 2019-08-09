"""
This attention-based seq2seq model is based on:
https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
"""
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
