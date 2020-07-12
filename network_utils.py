import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from utils import norm, denorm

def _vgg(output_layer):
    global hr_shape
    vgg = VGG19(input_shape = (None, None, 3), weights='imagenet', include_top=False)
    vgg.layers[output_layer].activation=None
    return Model(vgg.input, vgg.layers[output_layer].output)

def get_errors():
    mean_squared_error = tf.keras.losses.MeanSquaredError()
    mean_abs_error = tf.keras.losses.MeanAbsoluteError()
    binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return mean_squared_error, mean_abs_error, binary_cross_entropy

def generator_loss(binary_cross_entropy, sr_out):
    return binary_cross_entropy(tf.ones_like(sr_out), sr_out)

def discriminator_loss(binary_cross_entropy, hr_out, sr_out):
    real_y = tf.ones_like(hr_out) - tf.random.uniform(hr_out.shape)*0.2
    fake_y = tf.random.uniform(sr_out.shape)*0.2
    hr_loss = binary_cross_entropy(real_y , hr_out)
    sr_loss = binary_cross_entropy(fake_y, sr_out)
    return hr_loss + sr_loss

def l1_loss(mean_abs_error, hr, sr):
    return mean_abs_error(hr, sr)

@tf.function
def content_loss(vgg, mean_squared_error, hr, sr):
    sr = tf.keras.applications.vgg19.preprocess_input(sr)
    hr = tf.keras.applications.vgg19.preprocess_input(hr)
    sr_features = vgg(sr) / 12.75
    hr_features = vgg(hr) / 12.75
    return mean_squared_error(hr_features, sr_features)

def get_optimizer():
  schedule = PiecewiseConstantDecay(boundaries=[30000], values=[1e-4, 1e-5])
  adam = Adam(lr=schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  return adam

@tf.function
def train_step(lr, hr, gen, disc, generator_optimizer, discriminator_optimizer, vgg, mean_squared_error, mean_abs_error, binary_cross_entropy):
    """SRGAN training step.
    
    Takes an LR and an HR image batch as input and returns
    the computed perceptual loss and discriminator loss.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Forward pass
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)
            
        sr = gen(lr, training=True)
        hr_output = disc(hr, training=True)
        sr_output = disc(sr, training=True)
        # Compute losses
        con_loss = content_loss(vgg, mean_squared_error, hr, sr)
        gen_loss = generator_loss(binary_cross_entopy, sr_output)
        abs_loss = l1_loss(mean_abs_error, hr, sr)
        
        perc_loss = con_loss + 0.001 * gen_loss + 0.01 * abs_loss
        disc_loss = discriminator_loss(hr_output, sr_output)
        
    # Compute gradient of perceptual loss w.r.t. generator weights 
    gradients_of_generator = gen_tape.gradient(perc_loss, gen.trainable_variables)
    
    # Compute gradient of discriminator loss w.r.t. discriminator weights 
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)
    # Update weights of generator and discriminator
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    return perc_loss, disc_loss