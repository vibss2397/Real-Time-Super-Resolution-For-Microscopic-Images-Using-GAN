import numpy as np
import os
import imageio
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image

def patchify(im, patch_shape):
    res = np.array(im).shape
    patches = []
    for i in range(res[0]//patch_shape):
        x1 = patch_shape*i
        for j in range(res[1]//patch_shape):
            x2 = patch_shape*j
            img= im[x1:x1+patch_shape, x2:x2+patch_shape, :]
            patches.append(img)
        x2 = 0
    return np.array(patches)

def unpatchify(patches: np.ndarray, imsize: Tuple[int, int]):
    assert len(patches.shape) == 4
    nb_patches_w = imsize[1]//patches.shape[2]
    nb_patches_h = imsize[0]//patches.shape[1]
    img_final = np.zeros((patches.shape[1]*nb_patches_w, patches.shape[2]*nb_patches_h, 3))
    for i in range(nb_patches_w):
        x1 = patches.shape[1]*i
        for j in range(nb_patches_h):
            x2 = patches.shape[2]*j
            img_final[x1:x1+patches.shape[1], x2:x2+patches.shape[2], :] = patches[i*nb_patches_h+j] 
        x2 = 0
    return img_final

def name_to_image(filename, resolution):
    if(not resolution==1):
        i = Image.open(filename).resize((resolution[0], resolution[1]), Image.BICUBIC)
    else:
        i = Image.open(filename)
    a = np.array(i)
    return a
    
def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
    
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

def generate_train_batch(x_train, batch_size, rand_nums):
    train_hr = []
    train_lr = []
    
    for filename in rand_nums:
        img_patches_hr = patchify(name_to_image(x_train[filename], 1), 256)
        img_patches_lr = patchify(name_to_image(x_train[filename], (128, 128)), 64)
        for i in range(img_patches_hr.shape[0]):
          train_hr.append(img_patches_hr[i])
          train_lr.append(img_patches_lr[i])
    return normalize(np.array(train_hr)), normalize(np.array(train_lr))
  
def generate_test_batch(x_train, batch_size, rand_nums):
    train_hr = []
    train_lr = []
    for filename in rand_nums:
        train_hr.append(np.array(name_to_image(x_train[filename], 1)))
        train_lr.append(np.array(name_to_image(x_train[filename], (128, 128))))
    return normalize(np.array(train_hr)), normalize(np.array(train_lr))

def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for f in os.listdir(dirs): 
        if f.endswith(ext):
            file_names.append(os.path.join(dirs,f))
            count = count + 1
    return file_names 

def load_training_data(directory, ext, number_of_images = 1000, train_test_ratio = 0.8):
    number_of_train_images = int(number_of_images * train_test_ratio)
    files = load_data_from_dirs(directory, ext)
    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]
    return x_train, x_test, number_of_train_images

def generate_gif():
    with imageio.get_writer('./generate_test.gif', mode='I') as writer:
        filenames = glob.glob('./generated/generated_image_*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 14*(i**1.2)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

def plot_generated_images(output_dir, epoch, generator, x_test, dim=(1, 4), figsize=(35, 10)):
   
    examples = len(x_test)
    print(examples)
    
    rand_nums = np.random.randint(0, examples, size=32)
    x_test_hr, x_test_lr = generate_test_batch(x_test, 16, rand_nums )
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    value = randint(0, 32)
    img_target = image_batch_lr[value]
    patches = patchify(img_target, 64)
    print(patches.shape)
    gen_img = generator.predict(patches)
    generated_image = denormalize(gen_img)
    gen_image_unpatch = np.uint8(unpatchify(generated_image, (512, 512)))
    image_batch_lr = denormalize(image_batch_lr)
    
    plt.figure(figsize=figsize)
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(Image.fromarray(image_batch_lr[value]).resize((512, 512), Image.BICUBIC))
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(Image.fromarray(image_batch_lr[value]).resize((512, 512), Image.NEAREST))
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(gen_image_unpatch)
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 4)
    plt.imshow(image_batch_hr[value])
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_diff_%d.png' % epoch)
    plt.show()