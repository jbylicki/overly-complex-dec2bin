import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint
from model_def import build_model_func

model_filename = './model.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename,
    verbose=1,
    monitor='accuracy',
    save_best_only=True,
)

train_in = np.random.randint(0, 256, (100000, 1)).astype(np.uint8)
train_out = np.unpackbits(train_in, axis=1)

test_in = np.random.randint(0, 256, (1000, 1)).astype(np.uint8)
test_out = np.unpackbits(test_in, axis=1)

model = build_model_func()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="mean_squared_error",
    metrics=['accuracy']
)
model.fit(train_in, train_out, epochs=10, callbacks=[callback_checkpoint])
