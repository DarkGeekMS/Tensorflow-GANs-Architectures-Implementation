from base.base_model import BaseModel
import tensorflow as tf

class Generator:
    def __init__(self, depths=[1024,512,256,128], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, is_training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('gen', reuse=self.reuse):
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5,5], strides=(2,2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5,5], strides=(2,2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5,5], strides=(2,2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5,5], strides=(2,2), padding='SAME')
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        return outputs


class Discriminator:
    def __init__(self, depths=[64,128,256,512]):
        self.depths = [3] + depths
        self.reuse = False

    def __call__(self, inputs, is_training=False, name=''):
        raise NotImplementedError    


class DCGANModel(BaseModel):
    def __init__(self, config):
        super(DCGANModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        raise NotImplementedError

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)        