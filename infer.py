import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint


def binary_step(x):
    return tf.keras.activations.relu(x,alpha=0, max_value=1, threshold=0.5) 

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(1,1)),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'sigmoid'),
    tf.keras.layers.Dense(32, activation = 'sigmoid'),
    tf.keras.layers.Dense(16, activation = 'sigmoid'),
    tf.keras.layers.Dense(8, activation =tf.nn.sigmoid)
])
model.load_weights('./model.h5')
print((model.predict(np.array([[1],[2],[17]])) > 0.5).astype(np.uint8))
print((model.predict(np.array([[1],[2],[17]]))).astype(np.float))
