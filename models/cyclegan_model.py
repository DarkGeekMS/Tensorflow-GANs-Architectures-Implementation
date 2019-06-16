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
