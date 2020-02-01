# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:30:13 2020

@author: Georgios
"""

from keras.layers import Conv2D, Input, Add, LeakyReLU, DepthwiseConv2D, Lambda
from keras.initializers import RandomNormal
from keras.models import Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from subpixel import Subpixel, icnr_weights
from custom_initialisers import ICNR

import numpy as np
import scipy.stats as st
import tensorflow as tf

def G1(input_shape):
    
    def res_block(feature_vector_input):
        init = RandomNormal(stddev=0.02)
		
        y = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(feature_vector_input)
        y = LeakyReLU(alpha=0.2)(y)
		
        y = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(y)
        y = LeakyReLU(alpha=0.2)(y)
		
        return Add()([feature_vector_input, y])
    
    #define initialiser
    init = RandomNormal(stddev=0.02)
    
    image = Input(input_shape)
    x = Conv2D(64, 7, strides = 1, padding = 'SAME', kernel_initializer = init)(image)
    x = LeakyReLU(alpha=0.2)(x)
    
    for i in range(6):
        x = res_block(x)
    
    x = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
	
    x = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(3, 7, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    
    return Model(inputs=image, outputs=x, name="G1")

def D1(input_shape):
    init = RandomNormal(stddev=0.02)
    image = Input(input_shape)
	
    x = Conv2D(64, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(image)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = InstanceNormalization(axis = -1, center = False, scale = False)(x)
    
    x = Conv2D(256, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = InstanceNormalization(axis = -1, center = False, scale = False)(x)
    
    x = Conv2D(512, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = InstanceNormalization(axis = -1, center = False, scale = False)(x)
    
    x = Conv2D(1, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    
    return Model(inputs = image, outputs = x, name="D1")
	
	
	

def G2(input_shape):
    
    def res_block(feature_vector_input):
        init = RandomNormal(stddev=0.02)
		
        y = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(feature_vector_input)
        y = LeakyReLU(alpha=0.2)(y)
		
        y = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(y)
        y = LeakyReLU(alpha=0.2)(y)
		
        return Add()([feature_vector_input, y])
    
    #define initialiser
    init = RandomNormal(stddev=0.02)
    
    image = Input(input_shape)
    x = Conv2D(64, 7, strides = 1, padding = 'SAME', kernel_initializer = init)(image)
    x = LeakyReLU(alpha=0.2)(x)
    
    for i in range(6):
        x = res_block(x)
    
    x = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
	
    x = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(3, 7, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    
    return Model(inputs=image, outputs=x, name="G2")

def SR(input_shape):
    
    n_feats = 128
    n_resblocks = 8
    
    def res_block(input_tensor, num_filters, res_scale = 1.0):
        init = RandomNormal(stddev=0.02)
        x = Conv2D(num_filters, 3, strides = 1, padding='same', activation = 'relu', kernel_initializer = init)(input_tensor)
        x = Conv2D(num_filters, 3, strides = 1, padding='same', kernel_initializer = init)(x)
        x = Lambda(lambda x: x * res_scale)(x)
        x = Add()([x, input_tensor])
        return x
	
    inp = Input(shape = input_shape)
    
    init = RandomNormal(stddev=0.02)
    
    x = Conv2D(n_feats, 3, strides = 1, padding='same', kernel_initializer = init)(inp)
    conv1 = x
	
    if n_feats == 256:
        res_scale = 0.1
    else:
        res_scale = 1.0
		
    for i in range(n_resblocks): 
        x = res_block(x, n_feats, res_scale)
		
    x = Conv2D(n_feats, 3, strides = 1, padding='same', kernel_initializer = init)(x)
    x = Add()([x, conv1])
    
    
    x = Subpixel(n_feats, (3,3), 2, padding='same')(x)
	
    sr = Conv2D(3, 1, strides = 1, padding='same', kernel_initializer = init)(x)
    
    model = Model(inputs=inp, outputs=sr, name = 'SR')
    
    for layer in model.layers:
        if type(layer) == Subpixel:
            c, b = layer.get_weights()
            w = icnr_weights(scale=2, shape=c.shape)
            layer.set_weights([w, b])
    
    
    
    return model

def D2(input_shape):
    init = RandomNormal(stddev=0.02)
    
    image = Input(input_shape)
    
    x = Conv2D(64, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(image)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = InstanceNormalization(axis = -1, center = False, scale = False)(x)
    
    x = Conv2D(256, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = InstanceNormalization(axis = -1, center = False, scale = False)(x)
    
    x = Conv2D(512, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = InstanceNormalization(axis = -1, center = False, scale = False)(x)
    
    x = Conv2D(1, 4, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    
    return Model(inputs = image, outputs = x, name="D2")

def G3(input_shape):
    def res_block(feature_vector_input):
        init = RandomNormal(stddev=0.02)
		
        y = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(feature_vector_input)
        y = LeakyReLU(alpha=0.2)(y)
		
        y = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(y)
        y = LeakyReLU(alpha=0.2)(y)
		
        return Add()([feature_vector_input, y])
    
    #define initialiser
    init = RandomNormal(stddev=0.02)
    
    image = Input(input_shape)
    x = Conv2D(64, 7, strides = 1, padding = 'SAME', kernel_initializer = init)(image)
    x = LeakyReLU(alpha=0.2)(x)
    
	#downscaling operation
    x = Conv2D(64, 4, strides = 2, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    for i in range(5):
        x = res_block(x)
    
    x = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
	
    x = Conv2D(64, 3, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(3, 7, strides = 1, padding = 'SAME', kernel_initializer = init)(x)
    
    return Model(inputs = image, outputs = x, name="G3")

	
def blur(input_shape):
    def gauss_kernel(kernlen=21, nsig=3, channels=3):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter
    
    kernel_size=21
    blur_kernel_weights = gauss_kernel()
    
    image = Input(input_shape)
    g_layer = DepthwiseConv2D(kernel_size, use_bias=False, padding='same')
    image_processed = g_layer(image)
    
    g_layer.set_weights([blur_kernel_weights])
    g_layer.trainable = False
	
    return Model(inputs = image, outputs = image_processed, name="blur")
