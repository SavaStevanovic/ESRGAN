import tensorflow as tf


def session_initialiser():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)
