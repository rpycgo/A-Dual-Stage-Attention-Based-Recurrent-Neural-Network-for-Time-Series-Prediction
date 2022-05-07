from ..layer import NewLSTM, InputAttention, Encoder, TemporalAttention, Decoder

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Layer, Dense, RepeatVector, Permute, Lambda
from tensorflow.keras.models import Model


class DARNN(Model):
    def __init__(self, seq_len, **kwargs):
        super(DARNN, self).__init__(**kwargs)
        self.n = kwargs.get('n')
        self.encoder_dim = kwargs.get('encoder_dim')
        self.decoder_dim = kwargs.get('decoder_dim')
        self.encoder = Encoder(seq_len=seq_len, encoder_dim=self.encoder_dim)
        self.decoder = Decoder(seq_len=seq_len, encoder_dim=self.encoder_dim, decoder_dim=self.decoder_dim)
        self.dense1 = Dense(units=self.decoder_dim)
        self.dense2 = Dense(units=1)


    def call(self, x, training=False):
        encoder_data, decoder_data = x
        batch_size = encoder_data.shape[0]

        hidden_state = tf.zeros((batch_size, self.encoder_dim))
        cell_state = tf.clone(hidden_state)

        encoder_output = self.encoder(
            x=encoder_data,
            hidden_state=hidden_state,
            cell_state=cell_state,
            n=self.n,
            training=training
            )

        encoder_hidden_state = self.lstm(encoder_output)

        decoder_output = self.decoder(
            x=decoder_data,
            hidden_state=hidden_state,
            cell_state=cell_state,
            encoder_hidden_state=encoder_hidden_state,
            training=training
            )

        output = self.dense1(decoder_output)
        output = self.dense2(output)
        output = tf.squeeze(output)

        return output
