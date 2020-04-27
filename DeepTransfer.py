import tensorflow as tf
import numpy as np
import sys

import matplotlib.pyplot as plt
from PIL import Image

from model import *

# Download an image and read it into a NumPy array.
def get_image_as_array(file_name, size=200):
    img = Image.open(file_name)
    img = img.resize((size,size))
    img = img.convert('RGB')
    return np.array(img)


source = get_image_as_array("./Picasso.jpg")
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
style_names = ['block4_conv1']
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

plt.show()

def run_deep_transfer(source, target, steps=100, step_size=0.01):
    source = preprocess_inception(source)
    target = preprocess_inception(target)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = 100
        else:
          run_steps = steps_remaining
        steps_remaining -= run_steps
        step += run_steps

        # run_steps = tf.convert_to_tensor(run_steps)
        loss, img = deeptransfer(source, target, run_steps, step_size)


        plt.imshow(img)
        plt.show()
        print ("Step {}, loss {}".format(step, loss))


    #plt.imshow(img)
    #plt.show()

    return img

new_img = run_deep_transfer(source, target, steps=100, step_size=0.01)
