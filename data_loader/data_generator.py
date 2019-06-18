import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave, imresize
import os
import copy

class DataGenerator:
    def __init__(self, data_dir, resized_data_dir, t_size, batch_size):
        self.data_dir = data_dir
        self.resized_data_dir = resized_data_dir
        self.t_size = t_size
        self.batch_size = batch_size
        self.resize_imgs()
        self.load_data()

    def resize_imgs(self):
        img_list = []
        for img_name in (os.listdir(self.data_dir)):
            img_list.append(self.data_dir + img_name)
        i = 0
        for img in img_list:
            content = imread(img)
            content_resized = imresize(content, tuple(self.t_size[0:2]))
            imsave(self.resized_data_dir + "IMG_{}.JPG".format(i) , content_resized)
            i += 1

    def load_data(self):
        img_list = []
        for img_name in (os.listdir(self.resized_data_dir)):
            img_list.append(self.resized_data_dir + img_name)
        
        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string)
            image_resized = tf.image.resize_images(image_decoded, self.t_size[0:2])
            return image_resized
        
        filenames = tf.constant(img_list)
        dataset = tf.data.Dataset.from_tensor_slices((filenames))
        dataset = dataset.map(_parse_function)
        dataset = dataset.repeat().batch(self.batch_size)
        self.iter = dataset.make_one_shot_iterator()

    def next_batch(self):
        train_batch = self.iter.get_next()
        yield train_batch


class ImagePool:
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]     
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]  
        else:
            return image 