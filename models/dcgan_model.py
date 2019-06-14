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
        
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x*leak, name=name)

        outputs = tf.convert_to_tensor(inputs)
        with tf.name_scope('dis' + name), tf.variable_scope('dis', reuse=self.reuse):
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=is_training), name='outputs')
            with tf.variable_scope('fc'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
        return outputs
        

class DCGANModel(BaseModel):
    def __init__(self, config):
        super(DCGANModel, self).__init__(config)
        self.batch_size = self.config.batch_size
        self.s_size = self.config.s_size
        self.z_dim = self.config.z_dim
        self.gen = Generator(depths=self.config.g_depths, s_size=self.config.s_size)
        self.dis = Discriminator(depths=self.config.d_depths)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.train_data = tf.placeholder(tf.float32, shape=[None] + self.config.t_size)
        generated = self.gen(self.z, is_training=True)
        g_outputs = self.dis(generated, is_training=True, name='g')
        t_outputs = self.dis(self.train_data, is_training=True, name='t')
        tf.add_to_collection(
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=t_outputs)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        self.g = tf.add_n(tf.get_collection('g_losses'), name='total_g_loss')
        self.d = tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        g_opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.5) 
        d_opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.5)
        g_opt_op = g_opt.minimize(self.g, var_list=self.gen.variables, global_step=self.global_step_tensor)
        d_opt_op = d_opt.minimize(self.d, var_list=self.dis.variables, global_step=self.global_step_tensor)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            self.train_step = tf.no_op(name='train')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = self.gen(inputs, is_training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))                    