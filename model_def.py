import tensorflow as tf


def binary_step(x):
    return tf.keras.activations.relu(x, alpha=0, max_value=1, threshold=0.5)


def build_model_func():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 1)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(32, activation='sigmoid'),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(8, activation=tf.nn.sigmoid)
    ])
