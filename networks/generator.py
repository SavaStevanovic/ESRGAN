import tensorflow as tf


def build_generator(tf_x_image, tf_training, block_count=23, upscale_times=2):
    with tf.variable_scope("srgan"):
        print('\nBuilding first layer')
        net = tf.layers.conv2d(inputs=tf_x_image, filters=64, kernel_size=(
            3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        post_res = net

        print('\nBuilding block layers')
        with tf.variable_scope("RRDB_blocks"):
            for i in range(block_count):
                net = RRDB(i, net)

        print('\nBuilding pre upscale layer')
        net = tf.layers.conv2d(inputs=net, filters=64,
                               kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        net += post_res

        print('\nBuilding upscale layers')
        for i in range(upscale_times):
            with tf.variable_scope("upscale_layer_"+str(i)):
                net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(
                    3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.depth_to_space(net, block_size=2)
                net = tf.nn.leaky_relu(net)

        net = tf.layers.conv2d(
            inputs=net, filters=3, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(
            inputs=net, filters=3, kernel_size=(3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.identity(net, name='output_image')

        return net


def RRDB(i, net):
    net_pre = net
    with tf.variable_scope("RRDB_"+str(i)):
        with tf.variable_scope("dense_blocks_1_" + str(i)):
            net = dense_block(i, net)
        with tf.variable_scope("dense_blocks_2_" + str(i)):
            net = dense_block(i, net)
        with tf.variable_scope("dense_blocks_3_" + str(i)):
            net = dense_block(i, net)

        net = net * 0.2 + net_pre
        return net


def dense_block(i, net):
    with tf.variable_scope("dense_block_"+str(i)):
        net1 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(
            3, 3), padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        net2 = tf.layers.conv2d(inputs=tf.concat([net1, net], 3), filters=64, kernel_size=(
            3, 3), padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        net3 = tf.layers.conv2d(inputs=tf.concat([net2, net1, net], 3), filters=64, kernel_size=(
            3, 3), padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        net4 = tf.layers.conv2d(inputs=tf.concat([net3, net2, net1, net], 3), filters=64, kernel_size=(
            3, 3), padding='SAME', activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

        net5 = tf.layers.conv2d(inputs=tf.concat([net4, net3, net2, net1, net], 3), filters=64, kernel_size=(
            3, 3), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())

    return net5 * 0.2 + net
