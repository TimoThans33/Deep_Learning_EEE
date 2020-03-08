Training data error: 23.90 MSE
Test data error: 51.02 MSE
window size: 1

architecture:
rnn = Sequential()
rnn.add(LSTM(16, input_shape = (window_size, 1)))
rnn.add(Dense(1))
