import tensorflow as tf

import data_loader.image_loader as il
from models.ESRGAN import ESRGAN
from trainers.ESRGANtrainer import ESRGANTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from utils.session import session_initialiser


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config("C:\\Users\\Sava\\Documents\\ESRGAN\\configs\\config.json")

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = session_initialiser()
    # create your data generator
    data = il.ImageLoader(config)
    # create an instance of the model you want
    model = ESRGAN(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ESRGANTrainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
