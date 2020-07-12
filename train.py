import numpy as np
from numpy import array
from numpy.random import randint
from tqdm import tqdm
from typing import Tuple

from utils import *
from gan_architecture import *
from network_utils import *

channels = 3
image_shape = 512
lr_height = 64             # Low resolution height
lr_width = 64             # Low resolution width
lr_shape = (lr_height, lr_width,3)
hr_height = lr_height*4   # High resolution height
hr_width = lr_width*4     # High resolution width
hr_shape = (hr_height, hr_width, 3)
nb_patches = (image_shape/hr_width)**2
BATCH_SIZE = 2

def train(epochs, batch_size, input_dir, output_dir, number_of_images, train_test_ratio, downscale_factor = 4):
    global channels, lr_shape, hr_shape
    x_train, x_test, number_of_train_images = load_training_data(input_dir, '.png', number_of_images, train_test_ratio)  
    vgg = _vgg(20)
    batch_count = int(number_of_images / batch_size)
    shape = lr_shape
    gen = Generator(lr_shape).generator(64, False, 'interpolate')
    disc = Discriminator(hr_shape).discriminator(True)
    
    generator_optimizer = get_optimizer()
    discriminator_optimizer = get_optimizer()
    
    checkpoint_dir = 'models-bn/'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=gen,
                                    discriminator=disc)

    mean_sq_error, mean_abs_error, binary_crossentropy = get_errors()
    
    pls_metric = tf.keras.metrics.Mean()
    dls_metric = tf.keras.metrics.Mean()

    for e in range(epochs):
    print('epoch ', e)
    for i in tqdm(range(1000)):
        lr_image, hr_image = generate_train_batch(x_train, BATCH_SIZE, number_of_train_images)
        pl, dl = train_step(lr_image, hr_image, gen, disc, generator_optimizer, discriminator_optimizer, 
                                vgg, mean_sq_error, mean_abs_error, binary_crossentropy)
        pls_metric(pl)
        dls_metric(dl)
        if(i%200==0):
            print(f'{i}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
            
            loss_file = open(model_save_dir + 'losses2.txt' , 'a+')
            loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, pls_metric.result(), dls_metric.result()) )
            loss_file.close()
            
            pls_metric.reset_states()
            dls_metric.reset_states()
        if(i%500==0):
            plot_generated_images(output_dir, str(i)+' '+str(e), gen, x_test)
            print('saving img...')
    if(e%2==0):
        checkpoint.save(file_prefix = checkpoint_prefix)
        display.clear_output(wait=True)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    train(3000, 3, 'data', './generated/', './models/saved/', 'number_of_images_in_dataset', 'train_test_ratio')

