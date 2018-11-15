import cv2
import os
import os.path
from operator import itemgetter
import random
from sklearn.feature_extraction import image
import numpy as np


class ImageLoader:
    def __init__(self, config):
        self.shrink = config.shrink
        self.patch_size = config.patch_size
        self.batch_size = config.batch_size
        # specify your path here
        self.imageDir = config.image_dir
        # specify your vald extensions here
        self.valid_image_extensions = [
            ".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        self.image_path_list = []
        for file in os.listdir(self.imageDir):
            extension = os.path.splitext(file)[1]
            if extension.lower() not in self.valid_image_extensions:
                continue
            self.image_path_list.append(os.path.join(self.imageDir, file))
        self.batch_count = len(self.image_path_list)

    def shuffle_data(self):
        random.shuffle(self.image_path_list)

    def image_iterator(self):
        for i in range(self.batch_count):
            images = cv2.imread(self.image_path_list[i])
            prepreocessed_images = self.preprocess_images(images)
            if len(prepreocessed_images[0]) > 0 and len(prepreocessed_images[1]) > 0:
                yield prepreocessed_images

    def preprocess_images(self, images):
        try:
            if self.shrink:
                images = image.extract_patches_2d(image=images, patch_size=(
                    self.patch_size, self.patch_size), max_patches=self.batch_size)
                images = [cv2.flip(img, 1) if random.randint(0, 1) == 0 else img for img in images]
                images = [np.rot90(img, random.randint(0, 3)) for img in images]
            input_images = [cv2.resize(
                img, dsize=(img.shape[1]//4, img.shape[0]//4), interpolation=cv2.INTER_AREA) for img in images]
            images = [img/127.5-1 for img in images]
            input_images = [img/255 for img in input_images]
            return input_images, images
        except:
            return [], []