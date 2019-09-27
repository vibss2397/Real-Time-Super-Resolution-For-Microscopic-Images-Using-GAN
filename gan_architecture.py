from keras.layers import  Dense, Reshape, Flatten, Dropout, Concatenate, Input, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, add, Lambda
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model

def res_block(model, kernel_size, filters, strides):
    gen = model    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)        
    model = Add() ([gen, model])    
    return model

def res_block_without_bn(model, kernel_size, filters, strides):
    gen = model    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = Add() ([gen, model])    
    return model

def upsampling_transpose_block(model, kernel_size, filters, strides):
    model = Conv2DTranspose(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)    
    return model

def SubpixelConv2D(scale):
    import tensorflow as tf
    return Lambda(lambda x: tf.depth_to_space(x, scale))

def subpixel_upsampling_block(model, kernel_size, filters, strides, number):
  model = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', name='upSample_Conv2d_'+str(number))(model)
  model = SubpixelConv2D(2)(model)
  model = PReLU(shared_axes=[1,2], name='upSamplePReLU_'+str(number))(model)
  return model

def resize_like(scale): # resizes input tensor wrt. ref_tensor
    import tensorflow as tf
    return Lambda(lambda inputs: tf.keras.backend.resize_images(inputs, scale, scale, "channels_last", "nearest"))

def upsampling_then_conv_block(model, kernal_size, filters, strides, scale):
    model = resize_like(scale)(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    return model

def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = LeakyReLU(alpha = 0.2)(model)
    model = BatchNormalization(momentum = 0.5)(model)
    return model

class Generator(object):

    def __init__(self, input_shape):

      self.input_shape = input_shape

  def generator(self, use_bn, upsampling_type):
    gen_input = Input(shape = self.input_shape)
    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    gen_model = model
    # Using 16 Residual Blocks
    if(use_bn==True):
        for index in range(16):
            model = res_block(model, 3, 64, 1)
    elif(use_bn==False):
        for index in range(16):
            model = res_block_without_bn(model, 3, 64, 1)

    model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = add([gen_model, model])
    # Using 2 UpSampling Blocks
    if(upsampling_type=='srgan'):
        for index in range(2):
            model = subpixel_upsampling_block(model, 3, 256, 1, i)      
    elif(upsampling_type=='transpose'):
        for index in range(2):
            model = upsampling_transpose_block(model, 3, 256, 1)
    else:
        for index in range(2):
            model = upsampling_then_conv_block(model, 3, 256, 1, 2)
    model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)
    generator_model = Model(inputs = gen_input, outputs = model)
    return generator_model

class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self, use_GAP):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 64, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        if(use_GAP==True):      
            model = GlobalAveragePooling2D()(model)
        else:
            model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model




