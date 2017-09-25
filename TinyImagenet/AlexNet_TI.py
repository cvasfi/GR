#As designed on http://cs231n.stanford.edu/reports/2015/pdfs/lucash_final.pdf
import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
from LookupConvolution2d import lookup_conv2d


class model(object):

    def __init__(self,num_classes,mode,batch_size):
        print("alexNet tiny imagenet")
        self._extra_train_ops = []
        self.num_classes=num_classes
        self.mode=mode
        self.batch_size=batch_size
        self.global_step=tf.contrib.framework.get_or_create_global_step()

    def build(self,x,labels):
        input=tf.identity(x,"input")

        with tf.variable_scope('C1'):
            x=self.conv_relu(input,3,96)
        with tf.variable_scope('P1'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('C2'):
            x=tf.pad(x,[[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
            x=self.conv_relu(x,5,256)
        with tf.variable_scope('P2'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('C3'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.conv_relu(x,3,384)
        with tf.variable_scope('C4'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.conv_relu(x,3,384)
        with tf.variable_scope('C5'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.conv_relu(x,3,256)
        with tf.variable_scope('P3'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('FC1'):
            x=self.fc(x,4096)
            x=self.pRelu(x)
            x=tf.nn.dropout(x,0.5)
        with tf.variable_scope('FC2'):
            x=self.fc(x,4096)
            x=self.pRelu(x)
            x=tf.nn.dropout(x,0.5)
        with tf.variable_scope('FC3'):
            x=self.fc(x,200)

        self.logits=x
        self.predictions = tf.nn.softmax(self.logits,name="softmax")
        with tf.variable_scope('costs'):
          xent = tf.nn.softmax_cross_entropy_with_logits(
              logits=self.logits, labels=labels)
          self.cost = tf.reduce_mean(xent, name='xent')
          #self.cost += self._decay()

    def build_dw(self,x,labels):
        print("dw_separable")
        input=tf.identity(x,"input")
        with tf.variable_scope('C1'):
            x=self.conv_relu(x,3,96)
        with tf.variable_scope('P1'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('C2'):
            x=tf.pad(x,[[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
            x=self.dw_conv_relu(x,5,256)
        with tf.variable_scope('P2'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('C3'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.dw_conv_relu(x,3,384)
        with tf.variable_scope('C4'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.dw_conv_relu(x,3,384)
        with tf.variable_scope('C5'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.dw_conv_relu(x,3,256)
        with tf.variable_scope('P3'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('FC1'):
            x=self.fc(x,4096)
            x=self.pRelu(x)
            x=tf.nn.dropout(x,0.5)
        with tf.variable_scope('FC2'):
            x=self.fc(x,4096)
            x=self.pRelu(x)
            x=tf.nn.dropout(x,0.5)
        with tf.variable_scope('FC3'):
            x=self.fc(x,200)

        self.logits=x
        self.predictions = tf.nn.softmax(self.logits,name="softmax")
        with tf.variable_scope('costs'):
          xent = tf.nn.softmax_cross_entropy_with_logits(
              logits=self.logits, labels=labels)
          self.cost = tf.reduce_mean(xent, name='xent')

    def build_lookup(self,x,labels):
        input=tf.identity(x,"input")

        with tf.variable_scope('C1'):
            x=self.l_conv_relu(input,3,96)
        with tf.variable_scope('P1'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('C2'):
            x=tf.pad(x,[[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
            x=self.l_conv_relu(x,5,256)
        with tf.variable_scope('P2'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('C3'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.l_conv_relu(x,3,384)
        with tf.variable_scope('C4'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.l_conv_relu(x,3,384)
        with tf.variable_scope('C5'):
            x=tf.pad(x,[[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
            x=self.l_conv_relu(x,3,256)
        with tf.variable_scope('P3'):
            x=tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='VALID')
        with tf.variable_scope('FC1'):
            x=self.fc(x,4096)
            x=self.pRelu(x)
            x=tf.nn.dropout(x,0.5)
        with tf.variable_scope('FC2'):
            x=self.fc(x,4096)
            x=self.pRelu(x)
            x=tf.nn.dropout(x,0.5)
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
        x= tf.nn.conv2d(x, weights, strides, padding='VALID')
        return self.pRelu(x)

    def dw_conv_relu(self,x,ksize,outs,strides=[1,1,1,1]):
        ins = x.get_shape()[-1]
        n = ksize * ksize * 1
        dw_outs = 1  # depth multiplier: 1 filter per channel
        with tf.variable_scope("depthwise"):
            dw_weights = tf.get_variable(
                'DW', [ksize, ksize, ins, dw_outs],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            dws = tf.nn.depthwise_conv2d(x, dw_weights, strides, padding='VALID')
            dws = self.pRelu(dws)
            #
        with tf.variable_scope("pointwise"):
            return self.conv_relu(dws, 1, outs, strides=[1, 1, 1, 1])

    def l_conv_relu(self,x,ksize,outs, dict_size=30, i_sparsity=0.01, param_lambda=1.0, strides=1):
        x=lookup_conv2d(x, num_outputs=outs, kernel_size=[ksize,ksize], stride=strides, dict_size=dict_size,
                        padding=1,param_lambda=param_lambda,initial_sparsity=i_sparsity)
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

           return tf.nn.xw_plus_b(x, w, b)

           return x
