from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose

def res_block(x_in, num_filters, momentum=0.8, use_bn=False):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    if(use_bn):
        x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    if(use_bn):
        x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x

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

    def generator(self, num_filters, use_bn, upsampling_type):

        x_in = Input(shape=self.input_shape)
        x = Lambda(norm)(x_in)
        x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
        x = x_1 = PReLU(shared_axes=[1, 2])(x)

        for _ in range(16):
            if(use_bn):
                x = res_block(x, num_filters, True)
            else:
                x = res_block(x, num_filters, False)
    
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization(momentum=0.5)(x)
        x = Add()([x_1, x])
        if(upsampling_type=='srgan'):
            for index in range(2):
            model = subpixel_upsampling_block(model, 3, 256, 1, i)
        elif(upsampling_type=='transpose'):
            for index in range(2):
                model = upsampling_transpose_block(model, 3, 256, 1)
        else:
            x = up_sampling_block(x, 3, num_filters * 4, 1, 2)
            x = up_sampling_block(x, 3, num_filters * 4, 1, 2)

        x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
        x = Lambda(denorm)(x)
        return Model(x_in, x)
        
class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self, use_GAP):
        dis_input = Input(shape = self.image_shape)
        model = Lambda(norm)(dis_input)
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
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
        model = Dense(1, activation='sigmoid')(model)
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model