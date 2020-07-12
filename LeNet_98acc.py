import matplotlib.pyplot as plt
import pandas as pd
import keras
from skimage.transform import resize
from skimage import io
import glob
import cv2
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#define the convnet 
class LeNet:
	@staticmethod
	def build(input_shape, classes, DROPOUT=.1):
		model = Sequential()
		# CONV => RELU => POOL
		model.add(Conv2D(8, kernel_size=5,  padding="same",kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
			input_shape=input_shape))
		model.add(Activation("relu"))
		#model.add(Dropout(.3))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# CONV => RELU => POOL
		#model.add(Conv2D(16, kernel_size=5, padding="same"))
		#model.add(Activation("relu"))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(.1))
		# a softmax classifier
		model.add(Dense(1))
		model.add(Activation("sigmoid"))
		#model.add(BatchNormalization())
		#model.add(Dropout(.1))

		return model

if __name__ == '__main__':
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    print(x_train.shape, y_train.shape)
    np.random.seed(1671)# for reproducibility
    # network and training
    NB_EPOCH = 3
    BATCH_SIZE = 8
    VERBOSE = 1
    OPTIMIZER = Adam(lr=0.00001)#lr=0.00001)
    VALIDATION_SPLIT = 0.2
    IMG_ROWS, IMG_COLS = 200, 200 # input image dimensions
    NB_CLASSES = 2  # number of outputs
    #x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=.1, shuffle=True)
    #print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)
    y_train = y_train.reshape((-1, 1 ))
    x_train = x_train.reshape((-1, IMG_ROWS, IMG_COLS, 3 ))
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)
    model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER, metrics=['acc'])
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, shuffle=True)
