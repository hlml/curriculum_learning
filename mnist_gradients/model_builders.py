import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Activation, Reshape, Dropout, MaxPooling2D, BatchNormalization, Conv2D
from keras import backend as K
from keras.models import Model
import numpy as np
import re
from utils import variable_summaries, tf_get_uninitialized_variables

def build_dense_model(x, layers, num_class, batchnorm = False, bnmomentum = 0.99):
    num_layers = len(layers)
    layer_names = ["dense_layer" + str(i) for i in range(1, num_layers+1)]
    
    with tf.name_scope(layer_names[0]):
        xx = Dense(layers[0], kernel_initializer='glorot_normal')(x)
        if batchnorm:
            xx = BatchNormalization(momentum = bnmomentum)(xx, training=K.learning_phase())
        xx = Activation('relu')(xx)
            
    for l in range(1, num_layers):
        with tf.name_scope(layer_names[l]):
            xx = Dense(layers[l], kernel_initializer='glorot_normal')(xx)
            if batchnorm:
                xx = BatchNormalization(momentum = bnmomentum)(xx, training=K.learning_phase())
            xx = Activation('relu')(xx)

    with tf.name_scope('prob'):
        logits = Dense(num_class, kernel_initializer='glorot_normal')(xx)
    
    model = Model(inputs=x, outputs=logits)
    
    return model


def build_conv_model(x_image, layers, nodes, num_class, batchnorm = False, bnmomentum = 0.99, conv_kernel_size = (5,5), pool_size = (2,2)):
    num_layers = len(layers)
    layer_names = [layers[i-1] + str(i) for i in range(1, num_layers+1)]
    
    #add first layer, must be convolutional
    with tf.name_scope(layer_names[0]):
        if layers[0] == 'conv':
            xx = Conv2D(nodes[0], 
                           kernel_size=conv_kernel_size, 
                           strides=(1, 1), 
                           kernel_initializer='glorot_normal')(x_image)
            if batchnorm:
                xx = BatchNormalization(momentum = bnmomentum)(xx, training=K.learning_phase())
            xx = Activation('relu')(xx)
            
    
    #add subsequent layers
    j=1
    for name in layer_names[1:]:
        with tf.name_scope(name):
            if layers[j] == 'conv':
                xx = Conv2D(nodes[j], 
                           kernel_size=conv_kernel_size, 
                           strides=(1, 1), 
                           kernel_initializer='glorot_normal')(xx)
                if batchnorm:
                    xx = BatchNormalization(momentum = bnmomentum)(xx, training=K.learning_phase())
                xx = Activation('relu')(xx)
            elif layers[j] == 'maxpool':
                if layers[j+1] == 'dense':
                    xx = MaxPooling2D(pool_size=pool_size, strides=(1, 1))(xx)
                else:
                    xx = MaxPooling2D(pool_size=pool_size, strides=(2, 2))(xx)
            elif layers[j] == 'dense':
                if layers[j-1] != 'dense':
                    xx = Flatten()(xx)
                xx = Dense(nodes[j], kernel_initializer='glorot_normal')(xx)
                if batchnorm:
                    xx = BatchNormalization(momentum = bnmomentum)(xx, training=K.learning_phase())
                xx = Activation('relu')(xx)
            j = j + 1
    with tf.name_scope('prob'):
        logits = Dense(num_class, kernel_initializer='glorot_normal')(xx)
    
    model = Model(inputs=x_image, outputs=logits)
    
    return model
    
    
