from tensorflow.keras.layers import Input, LSTM, Layer, Dense
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
