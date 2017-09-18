from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dropout


def _conv_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return PReLU()(conv)

    return f



class AlexNetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs):
        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input = Input(shape=input_shape)
        output = _conv_relu(filters=96, kernel_size=(3, 3), strides=(1, 1))(input)
        output = MaxPooling2D()(output)
        output = _conv_relu(filters=256, kernel_size=(5, 5), strides=(1, 1))(output)
        output = MaxPooling2D()(output)
        output = _conv_relu(filters=384, kernel_size=(3, 3), strides=(1, 1))(output)
        output = _conv_relu(filters=385, kernel_size=(3, 3), strides=(1, 1))(output)
        output = _conv_relu(filters=256, kernel_size=(3, 3), strides=(1, 1))(output)
        output = MaxPooling2D()(output)

        output = Flatten()(output)
        output = Dense(units=4096)(output)
        output = Dropout(rate=0.5)(output)
        output= PReLU()(output)

        output = Dense(units=4096)(output)
        output = Dropout(rate=0.5)(output)
        output= PReLU()(output)

        output = Dense(units=200)(output)

        model = Model(inputs=input, outputs=output)
        return model

    @staticmethod
    def build(input_shape, num_outputs):
        return AlexNetBuilder.build(input_shape, num_outputs)

