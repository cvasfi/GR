import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

activation = tf.nn.relu


class model(object):

    def __init__(self,num_classes,mode,batch_size):
        print("alexNet tiny imagenet")
        self.num_classes=num_classes
        self.mode=mode
        self.batch_size=batch_size
        self.global_step=tf.contrib.framework.get_or_create_global_step()

    def build(self,x,labels):
        input=tf.identity(x,"input")

        with tf.variable_scope('C1'):
            x=self.conv_relu(x,3,96)
        with tf.variable_scope('P1'):
            x=tf.nn.max_pool(x,2,[1,2,2,1])
        with tf.variable_scope('C2'):
            x=self.conv_relu(x,5,256)
        with tf.variable_scope('P2'):
            x=tf.nn.max_pool(x,2,[1,2,2,1])
        with tf.variable_scope('C3'):
            x=self.conv_relu(x,3,384)
        with tf.variable_scope('C4'):
            x=self.conv_relu(x,3,384)
        with tf.variable_scope('C5'):
            x=self.conv_relu(x,3,256)
        with tf.variable_scope('P3'):
            x=tf.nn.max_pool(x,2,[1,2,2,1])
        with tf.variable_scope('FC1'):
            x=self.fc(x,4096)
        with tf.variable_scope('FC2'):
            x=self.fc(x,4096)
        with tf.variable_scope('FC3'):
            x=self.fc(x,200)

        self.logits=x
        self.predictions = tf.nn.softmax(self.logits,name="softmax")
        with tf.variable_scope('costs'):
          xent = tf.nn.softmax_cross_entropy_with_logits(
              logits=self.logits, labels=labels)
          self.cost = tf.reduce_mean(xent, name='xent')
          #self.cost += self._decay()


    def conv_relu(self,x,ksize,outs,strides=[1,1,1,1]):
        n = ksize * ksize * outs
        ins= x.get_shape()[-1]

        weights=tf.get_variable(
          'DW', [ksize, ksize, ins, outs],
          tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        x= tf.nn.conv2d(x, weights, strides, padding='SAME')
        return self.pRelu(x)

    def pRelu(self,_x):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def fc(self, x, out_dim):
           x = tf.reshape(x, [self.batch_size, -1])
           w = tf.get_variable(
               'DW', [x.get_shape()[1], out_dim],
               initializer=tf.contrib.layers.xavier_initializer())
           b = tf.get_variable('biases', [out_dim],
                               initializer=tf.constant_initializer())
           print("orig shape: ")
           print(w.get_shape())
           print(b.get_shape())

           return tf.nn.xw_plus_b(x, w, b)

           return x
