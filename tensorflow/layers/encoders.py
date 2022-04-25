import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Layer, Dense, RepeatVector, Permute
from tensorflow.keras.models import Model


class NewLSTM(LSTM):
    def __init__(self, units, **kwargs):
        super(NewLSTM, self).__init__(**kwargs)
        self.lstm = LSTM(units=units, return_sequences=True, return_state=True)

    def call(self, x, training=False):
        _, hidden_state, cell_state = self.lstm(x)
        self.initial_state = [hidden_state, cell_state]

        return hidden_state, cell_state

    def reset_state(self, hidden_state, cell_state):
        self.initial_state = [hidden_state, cell_state]


class InputAttention(Layer):
    def __init__(self, units, **kwargs):
        super(InputAttention, self).__init__(**kwargs)
        self.w1 = Dense(units)
        self.w2 = Dense(units)
        self.v = Dense(1)

    def call(self, hidden_state, cell_state, x):
        query = tf.concat([hidden_state, cell_state], axis=-1)
        query = RepeatVector(x.shape[2])(query)
        
        x_permuted = Permute((2, 1))(x)

        score = tf.nn.tanh(self.w1(x_permuted) + self.w2(query))
        score = self.v(score)
        score = Permute((2, 1))(score)

        attention_weights = tf.nn.softmax(score, axis=-1)

        context_vector = attention_weights * x
        context_vector = tf.reduce_sum(context_vector, axis=-1)

        return context_vector, attention_weights
