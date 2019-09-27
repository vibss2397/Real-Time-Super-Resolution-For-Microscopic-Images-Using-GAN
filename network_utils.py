import numpy as np

import keras.backend as K
from keras.applications.vgg19 import VGG19

from keras.layers import Input
from keras.models import Sequential, Model
from keras.optimizers import Adam

from gan_architecture import *

class VGG_LOSS(object):
    def __init__(self, image_shape):
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):  
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
        return K.mean(K.square(model(y_true) - model(y_pred)))

def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)
    return gan

def get_optimizer():
  adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  return adam