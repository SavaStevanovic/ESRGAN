import tensorflow as tf


def build_discriminator(tf_image, tf_training, reuse):
    with tf.variable_scope("discriminator", reuse=reuse):
        print('Building descriminator')

        with tf.variable_scope("discriminator_64_1"):
            net = tf.layers.conv2d(inputs=tf_image, filters=64, kernel_size=(
                3, 3), activation=tf.nn.leaky_relu, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())

        with tf.variable_scope("discriminator_64_2"):
            net = tf.layers.conv2d(
                inputs=net, filters=64, kernel_size=(4, 4), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_128_1"):
            net = tf.layers.conv2d(
                inputs=net, filters=128, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_128_2"):
            net = tf.layers.conv2d(
                inputs=net, filters=128, kernel_size=(4, 4), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_256_1"):
            net = tf.layers.conv2d(
                inputs=net, filters=256, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_256_2"):
            net = tf.layers.conv2d(
                inputs=net, filters=256, kernel_size=(4, 4), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_512_1"):
            net = tf.layers.conv2d(
                inputs=net, filters=512, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_512_2"):
            net = tf.layers.conv2d(
                inputs=net, filters=512, kernel_size=(4, 4), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_512_3"):
            net = tf.layers.conv2d(
                inputs=net, filters=512, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("discriminator_512_4"):
            net = tf.layers.conv2d(
                inputs=net, filters=512, kernel_size=(4, 4), padding='SAME', strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.batch_normalization(net, training=tf_training)
            net = tf.nn.leaky_relu(net)

        with tf.variable_scope("flatten"):
            net = tf.reshape(net, shape=[-1, 512*4*4])

        with tf.variable_scope("dense_1"):
            net = tf.layers.dense(inputs=net, units=100,
                                  activation=tf.nn.leaky_relu)
        with tf.variable_scope("dense_2"):
            net = tf.layers.dense(inputs=net, units=1)
        return net
