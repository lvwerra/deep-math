
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import numpy as np

class LSTM_S2S:
    """
    This seq2seq implementation is largely copied from:
    https://keras.io/examples/lstm_seq2seq/
    """
    def __init__(self, num_encoder_tokens, num_decoder_tokens, latent_dim):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.latent_dim = latent_dim

    def get_model(self):
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = CuDNNLSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = CuDNNLSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                             initial_state=self.encoder_states)
        self.decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = self.decoder_dense(decoder_outputs)

        return Model([self.encoder_inputs,  self.decoder_inputs], decoder_outputs)

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
        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.decoder_dense(decoder_outputs)

        decoder_model = Model([self.decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

        # reverse the char -> id dictionary for decoding
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

        # get the results for each sequence in the list
        results = []
        for input_seq in input_seq_list:
            results.append(self.decode_sequence(input_seq,
                                                encoder_model, decoder_model,
                                                target_token_index, reverse_target_char_index, max_sequence_length))
        return results

    def decode_sequence(self, input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_sequence_length):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_sequence_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, len(target_token_index)))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


class LSTM_Simple:

    def __init__(self, num_tokens, latent_dim):
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim

    def get_model(self):
        # Define an input sequence and process it.
        self.lstm_inputs = Input(shape=(None, self.num_tokens))
        self.lstm = CuDNNLSTM(self.latent_dim, return_state=True, return_sequences=True)
        lstm_outputs, state_h, state_c = self.lstm(self.lstm_inputs)

        self.lstm_states = [state_h, state_c]

        self.dense = Dense(self.num_tokens, activation='softmax')
        lstm_outputs = self.dense(lstm_outputs)

        self.model = Model(self.lstm_inputs, lstm_outputs)

        return self.model

    def decode_sample(self, input_seq_list, target_token_index, max_sequence_length):

        # reverse the char -> id dictionary for decoding
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self.lstm(self.lstm_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self.dense(decoder_outputs)

        decoder_model = Model([self.lstm_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


        # get the results for each sequence in the list
        results = []
        for input_seq in input_seq_list:
            results.append(self.decode_sequence(input_seq, decoder_model,
                                                target_token_index,
                                                reverse_target_char_index,
                                                max_sequence_length))
        return results

    def decode_sequence(self, input_seq, decoder_model, target_token_index,
                        reverse_target_char_index, max_sequence_length):

        # initial state is zero
        states_value = [np.zeros((1, self.latent_dim)), np.zeros((1, self.latent_dim))]

        # feed in the whole input sequence except the last thinking step which output will be used as
        # input for first relevant output_tokens
        _, h, c = decoder_model.predict([input_seq[:,:-1,:]] + states_value)
        states_value = [h, c]
        target_seq = input_seq[:,-1:,:]

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_sequence_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, len(target_token_index)))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence