"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os.path
import time
import numpy as np
import keras_resnet
import keras_alexnet
import keras_input

import sys
sys.setrecursionlimit(10000)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'normal', 'train or test')
tf.app.flags.DEFINE_string('cnn', 'normal', 'normal or dws')
tf.app.flags.DEFINE_string('network', 'alexnet', 'alexnet or resnet.')

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger(FLAGS.network+'.csv')
model_saver= ModelCheckpoint(FLAGS.network+".h5")

batch_size = 50 if FLAGS.network=="resnet" else 100
nb_classes = 200
nb_epoch = 1000
data_augmentation = True

# input image dimensions
img_rows, img_cols = 64, 64
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = keras_input.load_images("Data",200)
print("finished loading stuff")
# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.
if os.path.isfile(FLAGS.network+".h5"):
    model = load_model(FLAGS.network+".h5")
    print("model loaded")
else:
    if(FLAGS.network=="resnet"):
        model = keras_resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
    if FLAGS.network=="alexnet":
        model= keras_alexnet.AlexNetBuilder.buildAlexnet((img_channels, img_rows, img_cols), nb_classes)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


print("accuracy:")
start_time = time.time()
print(model.evaluate(X_test, Y_test,batch_size=1))
duration = time.time() - start_time

print ("duration: " + str(duration))
