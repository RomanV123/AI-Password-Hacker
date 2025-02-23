#model_building.py
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense # type: ignore

def build_model(vocab_size, seq_length):
    """
    Builds and compiles a simple Keras model:
      - An embedding layer that converts integer indices to vectors
      - An LSTM layer to capture sequential dependencies
      - A Dense output layer with softmax activation for predicting the next character
    """
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(128))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model