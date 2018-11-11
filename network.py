
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Activation, Dense, Input, Concatenate


model = Sequential()

x = Input(shape=(32,))
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
