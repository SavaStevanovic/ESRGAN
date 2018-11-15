from base.base_model import BaseModel
from networks.discriminator import build_discriminator
from networks.generator import build_generator
import tensorflow as tf
from vgg import vgg19
import tensorlayer as tl
import os
import numpy as np


class ESRGAN(BaseModel):
    def __init__(self, config):
        super(ESRGAN, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        # placeholders
        x = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.config.channels], name='x')
        y = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, self.config.channels], name='y')
        is_training = tf.placeholder(
            dtype=tf.bool, name='is_training')

        # models
        output = build_generator(x, y)
        real_discriminator_logits = build_discriminator(y, is_training, False)
        fake_discriminator_logits = build_discriminator(x, is_training, True)

        # vgg
        output224 = tf.image.resize_images(
            output, size=[224, 224], method=0, align_corners=False)
        tf_y_image244 = tf.image.resize_images(
            y, size=[224, 224], method=0, align_corners=False)

        self.vgg, output_content = vgg19.Vgg19_simple_api(
            (output224+1)/2, reuse=False)
        _, target_content = vgg19.Vgg19_simple_api(
            (tf_y_image244+1)/2, reuse=True)

        # discriminator loss
        fake_logit = (fake_discriminator_logits -
                      tf.reduce_mean(real_discriminator_logits))
        real_logit = (real_discriminator_logits -
                      tf.reduce_mean(fake_discriminator_logits))
        discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(
            fake_logit), logits=fake_logit) + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logit), logits=real_logit))
        discriminator_loss_summ = tf.summary.scalar(
            tensor=discriminator_loss, name='discriminator_loss_summ')

        # adverserial loss
        gen_loss = self.config.train.gan_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(
            real_logit), logits=real_logit) + tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logit), logits=fake_logit))
        gen_loss_summ = tf.summary.scalar(
            tensor=gen_loss, name='gen_loss_summ')

        # pixel loss
        pixel_loss = self.config.train.pixel_weight * \
            tf.losses.absolute_difference(output, y)
        l1_loss_summ = tf.summary.scalar(
            tensor=pixel_loss, name='pixel_loss_summ')

        # content loss
        content_loss = self.config.train.feature_weight * tf.losses.absolute_difference(
            target_content.outputs, output_content.outputs)
        content_loss_summ = tf.summary.scalar(
            tensor=content_loss, name='content_loss_summ')

        # sample images
        tf.summary.image(tensor=output, max_outputs=6, name='genearated')
        tf.summary.image(tensor=y, max_outputs=6, name='original')

        # variables
        srgan_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='srgan')
        disc_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

        # batch norm ops
        discriminator_update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        # generator trainings
        pre_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate)
        pre_optimizer = pre_optimizer.minimize(
            pixel_loss, name='pretrain_op', var_list=srgan_variables, global_step=self.global_step_tensor)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.learning_rate)
        optimizer = optimizer.minimize(
            pixel_loss+content_loss+gen_loss, name='train_op', var_list=srgan_variables, global_step=self.global_step_tensor)

        # discriminator ops
        with tf.control_dependencies(discriminator_update_ops):
            disc_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.discriminator_learning_rate)
            disc_optimizer = disc_optimizer.minimize(
                discriminator_loss, name='train_op_disc', var_list=disc_variables)

        self.merged = tf.summary.merge_all()

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def load_vgg19(self, sess):
        if not os.path.isfile(self.config.files.vgg19_npy_path):
            print(
                "Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(self.config.files.vgg19_npy_path,
                      encoding='latin1').item()

        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, self.vgg)
