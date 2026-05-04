import tensorflow as tf
from keras import layers

model = tf.keras.Sequential([
    layers.Embedding(20_000, 32),
    layers.Conv1D(32, 5, activation="relu", padding="same"),
    layers.MaxPooling1D(2),
    layers.Dropout(0.5),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(1, activation="sigmoid")
])