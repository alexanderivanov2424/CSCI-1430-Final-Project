import tensorflow as tf
import numpy as np
import sys

import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from model import *

# Download an image and read it into a NumPy array.
def get_image_as_array(file_name, size=200):
    img = Image.open(file_name)
    img = img.resize((size,size))
    img = img.convert('RGB')
    return np.array(img)


source = get_image_as_array("./Flag.jpg")
target = get_image_as_array("./Ocean.jpg")

#base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
base_model.summary()

# Maximize the activations of these layers
dream_names = ['block1_pool']

# Style is a function of these layers' activations
style_names = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

style_names = ['block1_conv1']
# style_names = ['block1_conv1']
dream_layers = [base_model.get_layer(name).output for name in dream_names]
style_layers = [base_model.get_layer(name).output for name in style_names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=dream_layers)
# Style extraction model
style_model = tf.keras.Model(inputs=base_model.input, outputs=style_layers)


deeptransfer = DeepTransfer(dream_model, style_model)

def preprocess_inception(img):
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    return img


def random_patch(img, size):
    x = np.random.randint(0,img.shape[0]-size)
    y = np.random.randint(0,img.shape[1]-size)
    return img[x:x+size, y:y+size], x, y

def patch_match_loss(source_patch, target_patch, model):
    source_patch = preprocess_inception(source_patch)
    target_patch = preprocess_inception(target_patch)

    img_batch = tf.expand_dims(source_patch, axis=0)
    source_act = model(img_batch)
    if not type(source_act) is list:
        source_act = [source_act]

    img_batch = tf.expand_dims(target_patch, axis=0)
    target_act = model(img_batch)
    if not type(target_act) is list:
        target_act = [target_act]

    return tf.add_n([tf.reduce_mean(tf.square(source_act[i] - target_act[i])) for i in range(len(source_act))])

def match_patch(source_patch, target, samples, model):
    best_patch = random_patch(target, source_patch[0].shape[0])
    best_loss = patch_match_loss(source_patch[0], best_patch[0], model)
    for k in range(samples):
        test_patch = random_patch(target, source_patch[0].shape[0])
        test_loss = patch_match_loss(source_patch[0], test_patch[0], model)

        if test_loss < best_loss:
            best_loss = test_loss
            best_patch = test_patch
    return best_patch

def create_mask(x,y,r,shape):
    mask = np.zeros(shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (x-i)*(x-i) + (y-j)*(y-j) < r*r:
                mask[i,j] = 1
    return gaussian_filter(mask,sigma=5,order=0)

def run_local_transfer(source, target, size, k, model):
    result = preprocess_inception(target).numpy()

    for transfer in range(k):
        source_patch = random_patch(source, size)
        target_patch = match_patch(source_patch, target, 50, model)

        S = preprocess_inception(source_patch[0])
        T = preprocess_inception(target_patch[0])
        loss, img = deeptransfer(S, T, 30, .01)



        _, x, y = target_patch
        mask = create_mask(x+size/2,y+size/2,size/4,result.shape)

        delta = np.zeros(result.shape)
        delta[x:x+size, y:y+size] = (img - result[x:x+size, y:y+size])
        result += mask*delta

        plt.imshow(np.clip(result+.5,0,1))
        plt.draw()
        plt.pause(.01)
        plt.cla()

    result += .5

    plt.imshow(result)
    plt.show()

run_local_transfer(source, target, 30, 1000, style_model)
