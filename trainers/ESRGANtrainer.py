from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class ESRGANTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ESRGANTrainer, self).__init__(
            sess, model, data, config, logger)
        self.model.load_vgg19(sess)

    def train_epoch(self):
        self.data.shuffle_data()
        self.iterator = self.data.image_iterator()
        loop = tqdm(range(self.data.batch_count))
        for i in loop:
            cur_it = self.model.global_step_tensor.eval(self.sess)
            summ = self.train_step(cur_it)
            if (i % 100 == 0):
                self.logger.summarize(cur_it, summary=summ)
        self.model.save(self.sess)

    def train_step(self, cur_it):
        try:
            batch_x, batch_y = next(self.iterator)
        except:
            return None
        if not batch_x or not batch_y:
            return None
        feed_dict = {"x:0": batch_x, "y:0": batch_y, "is_training:0": True}

        if self.config.training_mode == "pretrain":
            _, summ = self.sess.run(["pretrain_op", self.model.merged],
                                    feed_dict=feed_dict)

        else:
            _, summ = self.sess.run(["train_op", self.model.merged],
                                    feed_dict=feed_dict)
            _, summ = self.sess.run(["train_op_disc", self.model.merged],
                                    feed_dict=feed_dict)
        return summ
