#!/usr/bin/python
## -*- coding: utf-8 -*-
#
# Copyright (C) 2018
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#Author: Fernando Rodrigo Aguilar Javier
#Author email: faguilar@comunidad.unam.mx

#Detalles del codigo --------------------------------------------------------------------
#Solo es necesario cambiar el campo de la variable path para que pandas encuentre el path
#--------------------------------------------------------
#--------------------------------------------------------
#GENERATE DATA FOR LOAD FROM NPY NUMPY ARRAY
#Total initial dataset: 
###########################
#Before
#length of Data set train: 33543
#Positive  Negative
#1962      31581
##################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%################################
# import the necessary packages
#Conda install matplotlib keras-gpu pandas scikit-learn ipython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import tensorflow as tf

class LeNet:
	@staticmethod
	def build(input_shape, classes, DROPOUT=.1):
		model = Sequential()
		# CONV => RELU => POOL
		model.add(Conv2D(8, kernel_size=5,  padding="same", input_shape=input_shape))
                        #kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1),
                        #input_shape=input_shape))
		model.add(Activation("relu"))
		#model.add(Dropout(.3))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# CONV => RELU => POOL
		model.add(Conv2D(16, kernel_size=5, padding="same"))
		model.add(Activation("relu"))
        #model.add(Dropout(.3))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # CONV => RELU => POOL
		#model.add(Conv2D(32, kernel_size=5, padding="same"))
		#model.add(Activation("relu"))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #model.add(Conv2D(16, kernel_size=5, padding="same"))
		#model.add(Activation("relu"))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(1000))
		model.add(BatchNormalization())
		model.add(Activation("relu"))
		#model.add(Dropout(.1))
		# a softmax classifier
		model.add(Dense(1))
		model.add(Activation("sigmoid"))
		#model.add(BatchNormalization())
		#model.add(Dropout(.1))
		return model

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    #plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    #plot_roc_curve(fpr, tpr)
    plt.show()


if __name__ == '__main__':
    # Constantes
    np.random.seed(1671)# for reproducibility
    # network and training
    n, m =  8135, 4088#10847, 4088
    NB_EPOCH = 50
    BATCH_SIZE = 64# bz_e 64_5 128_7 64_19
    VERBOSE = 1
    OPTIMIZER = Adam(lr=0.00001)#lr=0.00001)SGD, RMSprop, Adam
    VALIDATION_SPLIT = 0.1
    IMG_ROWS, IMG_COLS = 64, 64 # input image dimensions
    INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)
    NB_CLASSES = 2  # number of outputs
    # Load data
    X = list(np.load('x_benign64_4.npy')) + list(np.load('x_malign64.npy'))
    Y = np.array(n * [0] + m * [1])
    Y = Y.reshape((-1, 1 ))
    X = np.array(X).reshape((-1, IMG_ROWS, IMG_COLS, 3 ))
    #x_train, y_train = X, Y
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.1, random_state=1671)
    model = LeNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER, metrics=['acc'])
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, shuffle=True)
    score = model.evaluate(x_test, y_test, verbose=VERBOSE)
    print(score)
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    #plt.grid(True)
    #plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    #y_pred = model.predict(x_test).ravel()
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    #plot_roc_curve(fpr, tpr)
    plt.show()
    df = pd.DataFrame()
    x_test = np.load('x_test64.npy')
    y_pred = model.predict(x_test)
    df_test= pd.read_csv('test.csv')
    df['image_name'] = df_test['image_name']
    df['target'] = [i[0].round(1) for i in y_pred]
    df.to_csv('submmit1', index=None)
