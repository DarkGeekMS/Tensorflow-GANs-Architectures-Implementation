import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave, imresize
import os

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.resize_imgs()
        self.load_data()

    def resize_imgs(self):
        img_list = []
        for img_name in (os.listdir(self.config.data_dir)):
            img_list.append(self.config.data_dir + img_name)
        i = 0
        for img in img_list:
            content = imread(img)
            content_resized = imresize(content, tuple(self.config.t_size[0:2]))
            imsave(self.config.resized_data_dir + "IMG_{}.JPG".format(i) , content_resized)
            i += 1

    def load_data(self):
        img_list = []
        for img_name in (os.listdir(self.config.resized_data_dir)):
            img_list.append(self.config.resized_data_dir + img_name)
        
        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded, self.config.t_size[0:2])
            return image_resized
        
        filenames = tf.constant(img_list)
        dataset = tf.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.map(_parse_function)
        dataset = dataset.repeat().batch(self.config.batch_size)
        self.iter = dataset.make_one_shot_iterator()

    def next_batch(self):
        train_batch = self.iter.get_next()
        yield train_batch