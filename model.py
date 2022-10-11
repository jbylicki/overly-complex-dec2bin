import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint

def binary_step(x):
    return tf.keras.activations.relu(x,alpha=0, max_value=1, threshold=0.5) 
model_filename = './model.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='accuracy', 
    save_best_only=True,
)

train_in = np.random.randint(0,256, (100000,1)).astype(np.uint8)
train_out = np.unpackbits(train_in,axis=1)

test_in = np.random.randint(0,256, (1000,1)).astype(np.uint8)
test_out = np.unpackbits(test_in,axis=1)

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
model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error", metrics=['accuracy'])
model.fit(train_in, train_out, epochs = 10, callbacks=[callback_checkpoint])

