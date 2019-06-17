from base.base_model import BaseModel

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops

class Generator:
    def __init__(self, gf_dim, output_c_dim, name="generator"):
        self.gf_dim = gf_dim
        self.output_c_dim = output_c_dim
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            c0 = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            c1 = tf.nn.relu(self.instance_norm(self.conv2d(c0, self.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
            c2 = tf.nn.relu(self.instance_norm(self.conv2d(c1, self.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
            c3 = tf.nn.relu(self.instance_norm(self.conv2d(c2, self.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
            r1 = self.residual_block(c3, self.gf_dim*4, name='g_r1')
            r2 = self.residual_block(r1, self.gf_dim*4, name='g_r2')
            r3 = self.residual_block(r2, self.gf_dim*4, name='g_r3')
            r4 = self.residual_block(r3, self.gf_dim*4, name='g_r4')
            r5 = self.residual_block(r4, self.gf_dim*4, name='g_r5')
            r6 = self.residual_block(r5, self.gf_dim*4, name='g_r6')
            r7 = self.residual_block(r6, self.gf_dim*4, name='g_r7')
            r8 = self.residual_block(r7, self.gf_dim*4, name='g_r8')
            r9 = self.residual_block(r8, self.gf_dim*4, name='g_r9')
            d1 = self.deconv2d(r9, self.gf_dim*2, 3, 2, name='g_d1_dc')
            d1 = tf.nn.relu(self.instance_norm(d1, 'g_d1_bn'))
            d2 = self.deconv2d(d1, self.gf_dim, 3, 2, name='g_d2_dc')
            d2 = tf.nn.relu(self.instance_norm(d2, 'g_d2_bn'))
            d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            self.pred = tf.nn.tanh(self.conv2d(d2, self.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))                       

    def instance_norm(self, inputs, name="instance_norm"):
        with tf.variable_scope(name):
            depth = inputs.get_shape()[3]
            scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized_inputs = (inputs - mean) * inv
            return scale*normalized_inputs + offset  

    def conv2d(self, inputs, output_dim, kernel_size=4, strides=2, stddev=0.02, padding='SAME', name="conv2d"):
        with tf.variable_scope(name):
            return slim.conv2d(inputs, output_dim, kernel_size, strides, padding=padding, activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                biases_initializer=None)   

    def deconv2d(self, inputs, output_dim, kernel_size=4, strides=2, stddev=0.02, name="deconv2d"):
        with tf.variable_scope(name):
            return slim.conv2d_transpose(inputs, output_dim, kernel_size, strides, padding='SAME', activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                        biases_initializer=None)    

    def residual_block(self, inputs, dim, kernel_size=3, strides=1, name="res"):
        p = int((kernel_size - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = self.instance_norm(self.conv2d(y, dim, kernel_size, strides, padding='VALID', name=name+'_c1'), name+'_bn1')
        y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = self.instance_norm(self.conv2d(y, dim, kernel_size, strides, padding='VALID', name=name+'_c2'), name+'_bn2')
        return x + y      


class Discriminator:
    def __init__(self, df_dim, name="discriminator"):
        self.df_dim = df_dim
        self.name = name

    def __call__(self, inputs, reuse=False):
        with tf.variable_scope(name):
            if reuse: 
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            h0 = self.lrelu(self.conv2d(inputs, self.df_dim, name='d_h0_conv'))
            h1 = self.lrelu(self.instance_norm(self.conv2d(h0, self.df_dim*2, name='d_h1_conv'), 'd_bn1'))
            h2 = self.lrelu(self.instance_norm(self.conv2d(h1, self.df_dim*4, name='d_h2_conv'), 'd_bn2'))
            h3 = self.lrelu(self.instance_norm(self.conv2d(h2, self.df_dim*8, strides=1, name='d_h3_conv'), 'd_bn3')) 
            self.pred = self.conv2d(h3, 1, strides=1, name='d_pred')           

    def instance_norm(self, inputs, name="instance_norm"):
        with tf.variable_scope(name):
            depth = inputs.get_shape()[3]
            scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
            mean, variance = tf.nn.moments(inputs, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized_inputs = (inputs - mean) * inv
            return scale*normalized_inputs + offset  

    def conv2d(self, inputs, output_dim, kernel_size=4, strides=2, stddev=0.02, padding='SAME', name="conv2d"):
        with tf.variable_scope(name):
            return slim.conv2d(inputs, output_dim, kernel_size, strides, padding=padding, activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                biases_initializer=None)

    def lrelu(self, inputs, leak=0.2, name="lrelu"):
        return tf.maximum(inputs, leak * inputs)                                                                                              


class CycleGANModel(BaseModel):
    def __init__(self, config):
        super(CycleGANModel, self).__init__(config)
        self.batch_size = self.config.batch_size
        self.image_size = self.config.image_size
        self.input_c_dim = self.config.input_nc
        self.output_c_dim = self.config.output_nc
        self.l1_lambda = self.config.l1_lambda
        self.data_dir = self.config.data_dir
        self.dis_a = Discriminator(self.config.df_dim, name="discriminatorA")
        self.dis_b = Discriminator(self.config.df_dim, name="discriminatorB")
        self.gen_ab = Generator(self.config.gf_dim, self.output_c_dim, name="generatorA2B")
        self.gen_ba = Generator(self.config.gf_dim, self.output_c_dim, name="generatorB2A")
        self.loss = self.mae_criterion
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32, 
                                        [None, self.image_size, self.image_size, 
                                        self.input_c_dim + self.output_c_dim],
                                        name="real_images")
        self.real_A = self.real_data[:,:,:,:self.input_c_dim]
        self.real_B = self.real_data[:,:,:,self.input_c_dim:self.input_c_dim+self.output_c_dim]
        
        self.fake_B = self.gen_ab(self.real_A, reuse=False)
        self.fake_Acc = self.gen_ba(self.fake_B, reuse=False)
        self.fake_A = self.gen_ba(self.real_B, reuse=True)
        self.fake_Bcc = self.gen_ab(self.fake_A, reuse=True)

        self.DA_fake = self.dis_a(self.fake_A, reuse=False)
        self.DB_fake = self.dis_b(self.fake_B, reuse=False)

        self.g_loss_a2b = self.loss(self.DB_fake, tf.ones_like(self.DB_fake)) \
                        + self.l1_lambda * self.abs_criterion(self.real_A, self.fake_Acc) \
                        + self.l1_lambda * self.abs_criterion(self.real_B, self.fake_Bcc) 
        self.g_loss_b2a = self.loss(self.DA_fake, tf.ones_like(self.DA_fake)) \ 
                        + self.l1_lambda * self.abs_criterion(self.real_A, self.fake_Acc) \
                        + self.l1_lambda * self.abs_criterion(self.real_B, self.fake_Bcc)
        self.g_loss = self.loss(self.DA_fake, tf.ones_like(self.DA_fake)) \ 
                    + self.loss(self.DB_fake, tf.ones_like(self.DB_fake)) \
                    + self.l1_lambda * self.abs_criterion(self.real_A, self.fake_Acc) \
                    + self.l1_lambda * self.abs_criterion(self.real_B, self.fake_Bcc)
                    
        self.fake_A_sample = tf.placeholder(tf.float32, 
                            [None, self.image_size, self.image_size, self.input_c_dim], 
                            name="fake_A_sample")
        self.fake_B_sample = tf.placeholder(tf.float32,
                            [None, self.image_size, self.image_size, self.output_c_dim],
                            name="fake_B_sample")
        self.DB_real = self.dis_b(self.real_B, reuse=True)        
        self.DA_real = self.dis_a(self.real_A, reuse=True)
        self.DB_fake_sample = self.dis_b(self.fake_B_sample, reuse=True)
        self.DA_fake_sample = self.dis_a(self.fake_A_sample, reuse=True)

        self.db_loss_real = self.loss(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.loss(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.loss(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.loss(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]                                                                 

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def mae_criterion(self, inputs, target):
        return tf.reduce_mean((inputs-target)**2)

    def abs_criterion(self, inputs, target):
        return tf.reduce_mean(tf.abs(inputs-target))    

    def sample_images(self):
        raise NotImplementedError            
