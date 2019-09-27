import numpy as np
from numpy import array
from numpy.random import randint
from tqdm import tqdm
from typing import Tuple

from utils import *
from gan_architecture import *
from network_utils import *

channels = 3
lr_height = 64             # Low resolution height
lr_width = 64             # Low resolution width
lr_shape = (lr_height, lr_width,3)
hr_height = lr_height*4   # High resolution height
hr_width = lr_width*4     # High resolution width
hr_shape = (hr_height, hr_width, 3)

def train(epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio, downscale_factor = 4):
    global channels, lr_shape, hr_shape
    x_train, x_test, number_of_train_images = load_training_data(input_dir, '.png', number_of_images, train_test_ratio)  
    loss = VGG_LOSS(hr_shape) 
    batch_count = int(number_of_images / batch_size)
    shape = lr_shape
    generator = Generator(shape).generator(False, 'interpolate')
    discriminator = Discriminator(hr_shape).discriminator(True)

#     generator.load('./models/srgan+interpolation+improved_disc.hdf5')
#     discriminator.load ('drive/srgan/model-4/dis_model_dif_20.h5')
    optimizer = get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)

    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)

#     loss_file = open(model_save_dir + 'losses.txt' , 'w+')
#     loss_file.close()
    
    for e in range(epochs):
       
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(500)):
            
            rand_nums = np.random.randint(0, number_of_train_images, size=batch_size)
            
            
            image_batch_hr, image_batch_lr = generate_train_batch(x_train, batch_size, rand_nums)
            generated_images_sr = generator.predict(image_batch_lr)
            real_data_Y = np.ones(12) - np.random.random_sample(12)*0.2
            fake_data_Y = np.random.random_sample(12)*0.2
       
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            rand_nums = np.random.randint(0, number_of_train_images, size=batch_size)
            image_batch_hr, image_batch_lr = generate_train_batch(x_train, batch_size, rand_nums)
            
            gan_Y = np.ones(12) - np.random.random_sample(12)*0.2
            discriminator.trainable = False

            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
           
            
        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)
        
        loss_file = open(model_save_dir + 'losses2.txt' , 'a+')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )
        loss_file.close()

        if e == 1 or (e) % 5 == 0:
            plot_generated_images(output_dir, e, generator, x_test)
        if (e) % 20 == 0:
            generator.save(model_save_dir + 'gen_model_dif_%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model_dif_%d.h5' % e)

if __name__ == "__main__":
    train(3000, 3, 'data', './generated/', './models/saved/', 'number_of_images_in_dataset', 'train_test_ratio')

