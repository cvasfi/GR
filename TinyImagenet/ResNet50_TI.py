import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

activation = tf.nn.relu


class model(object):

    def __init__(self,num_classes,mode,batch_size):
        print("network: resnet50")
        self._extra_train_ops = []
        self.num_classes=num_classes
        self.mode=mode
        self.batch_size=batch_size
        self.global_step=tf.contrib.framework.get_or_create_global_step()

    def build(self,x,labels):
        print("build start: ")
        print(x.get_shape())
        input=tf.identity(x,"input")
        with tf.variable_scope('init'):
            x=self.conv_bn_relu(input,3,32,[1,1,1,1])   #128,56,56,32

        with tf.variable_scope('stack1'):
            for i in range(3):
                x=self.block(x,str(i),64,(i==0),False)   #128,56,56,16

        with tf.variable_scope('stack2'):
            for i in range(4):
                x=self.block(x,str(i),128,(i==0))   #128,28,28,16

        with tf.variable_scope('stack3'):
            for i in range(6):
                x=self.block(x,str(i),256,(i==0))   #128,14,14,16

        with tf.variable_scope('stack4'):
            for i in range(3):
                x=self.block(x,str(i),256,(i==0))   #128,7,7,16

        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        with tf.variable_scope('fc'):
            x=self.fc(x,200)

        self.logits=x
        self.predictions = tf.nn.softmax(self.logits,name="softmax")
        with tf.variable_scope('costs'):
          xent = tf.nn.softmax_cross_entropy_with_logits(
              logits=self.logits, labels=labels)
          self.cost = tf.reduce_mean(xent, name='xent')
          self.cost += self._decay()

    def build_dw(self,x,labels):
        print("resnet dw build start: ")
        print(x.get_shape())
        input=tf.identity(x,"input")
        with tf.variable_scope('init'):
            x=self.conv_bn_relu(input,3,32,[1,1,1,1])   #128,56,56,32

        with tf.variable_scope('stack1'):
            for i in range(3):
                x=self.block_dw(x,str(i),64,(i==0),False)   #128,56,56,16

        with tf.variable_scope('stack2'):
            for i in range(4):
                x=self.block_dw(x,str(i),128,(i==0))   #128,28,28,16

        with tf.variable_scope('stack3'):
            for i in range(6):
                x=self.block_dw(x,str(i),256,(i==0))   #128,14,14,16

        with tf.variable_scope('stack4'):
            for i in range(3):
                x=self.block_dw(x,str(i),256,(i==0))   #128,7,7,16

        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        with tf.variable_scope('fc'):
            x=self.fc(x,200)

        self.logits=x
        self.predictions = tf.nn.softmax(self.logits,name="softmax")
        with tf.variable_scope('costs'):
          xent = tf.nn.softmax_cross_entropy_with_logits(
              logits=self.logits, labels=labels)
          self.cost = tf.reduce_mean(xent, name='xent')
          self.cost += self._decay()

    def block(self,x,name,outs,project=False,downsample=True):
        shortcut=x
        strides=[1,2,2,1] if (project and downsample) else [1,1,1,1]
        with tf.variable_scope(name):
            with tf.variable_scope("B1"):
                x=self.conv_bn_relu(x,1,outs,strides)
            with tf.variable_scope("B2"):
                x=self.conv_bn_relu(x,3,outs)
            with tf.variable_scope("B3"):
                x = self.conv(x, 1, outs*4)
                x= self.bn(x)

            with tf.variable_scope('shortcut'):
                if project:
                    shortcut = self.conv(shortcut, 1, outs*4,strides)
                    shortcut = self.bn(shortcut)
            return activation(x+shortcut)

    def block_dw(self,x,name,outs,project=False,downsample=True):
        shortcut=x
        strides=[1,2,2,1] if (project and downsample) else [1,1,1,1]
        with tf.variable_scope(name):
            with tf.variable_scope("B1"):
                x=self.conv_bn_relu(x,1,outs,strides)
            with tf.variable_scope("B2"):
                x=self.dw_conv_bn_relu(x,3,outs)
            with tf.variable_scope("B3"):
                x = self.conv(x, 1, outs*4)
                x= self.bn(x)

            with tf.variable_scope('shortcut'):
                if project:
                    shortcut = self.conv(shortcut, 1, outs*4,strides)
                    shortcut = self.bn(shortcut)
            return activation(x+shortcut)

    def conv_bn_relu(self,x,ksize,outs,strides=[1,1,1,1]):
        x=self.conv(x,ksize,outs,strides=strides)
        x=self.bn(x)
        return activation(x)

    def dw_conv_bn_relu(self, x, ksize, outs, strides=[1, 1, 1, 1]):
        x = self.dw_conv(x, ksize, outs, strides=strides)
        x = self.bn(x)
        return activation(x)

    def dw_conv(self,x,ksize,outs,strides=[1,1,1,1]):
        ins = x.get_shape()[-1]
        n = ksize * ksize * 1
        dw_outs = 1  # depth multiplier: 1 filter per channel
        with tf.variable_scope("depthwise"):
            dw_weights = tf.get_variable(
                'DW', [ksize, ksize, ins, dw_outs],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            dws = tf.nn.depthwise_conv2d(x, dw_weights, strides, padding='SAME')
            dws=self.bn(dws)
            dws = activation(dws)
            #
        with tf.variable_scope("pointwise"):
            return self.conv(dws, 1, outs, strides=[1, 1, 1, 1])

    def conv(self,x,ksize,outs,strides=[1,1,1,1]):
        n = ksize * ksize * outs
        ins= x.get_shape()[-1]

        weights=tf.get_variable(
          'DW', [ksize, ksize, ins, outs],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
        return tf.nn.conv2d(x, weights, strides, padding='SAME')


    def bn(self,x):
        with tf.variable_scope('batch_normalization'):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def fc(self, x, out_dim):
           x = tf.reshape(x, [self.batch_size, -1])
           w = tf.get_variable(
               'DW', [x.get_shape()[1], out_dim],
               initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
           b = tf.get_variable('biases', [out_dim],
                               initializer=tf.constant_initializer())
           print("orig shape: ")
           print(w.get_shape())
           print(b.get_shape())

           return tf.nn.xw_plus_b(x, w, b)

           return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.multiply(0.0002, tf.add_n(costs))
