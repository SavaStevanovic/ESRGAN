from base.base_model import BaseModel
import tensorflow as tf


class TemplateModel(BaseModel):
    def __init__(self, config):
        super(TemplateModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        tf_x = tf.placeholder(
            shape=[None, None, None, self.config.channels], dtype=tf.float32, name='tf_x')
        tf_y = tf.placeholder(
            shape=[None, None, None, self.config.channels], dtype=tf.float32, name='tf_y')
        tf_y_onehot = tf.one_hot(tf_x, self.config.depth, name='tf_y_onehot')

        with tf.variable_scope("input_layer"):
            net = tf.layers.separable_conv2d(inputs=tf_x, padding='SAME', kernel_size=(
                9, 9), depthwise_initializer=tf.contrib.layers.xavier_initializer(), depth_multiplier=self.config.depth_multiplier, activation=tf.nn.relu)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        pass
