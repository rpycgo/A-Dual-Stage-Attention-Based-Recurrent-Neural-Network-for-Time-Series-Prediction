from tensorflow.keras.layers import Input, LSTM, Layer
from tensorflow.keras.models import Model


class EncoderLSTM(Layer):
    def __init__(self, units):
        super(EncoderLSTM, self).__init__(name='encoder_lstm')
        self.lstm = LSTM(m, return_sequences=True, return_state=True)

    def call(self, x, training=False):
        _, hidden_state, cell_state = self.lstm(x)
        self.initial_state = [hidden_state, cell_state]

        return hidden_state, cell_state
    
    def reset_state(self, hidden_state, cell_state):
        self.initial_state = [hidden_state, cell_state]
