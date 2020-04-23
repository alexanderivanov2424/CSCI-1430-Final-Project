import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from model import *

# Download an image and read it into a NumPy array.
def get_image_as_array(file_name, size=200):
    img = Image.open(file_name)
    img = img.resize((size,size))
    return np.array(img)


source = get_image_as_array("./Flag.png")
target = get_image_as_array("./Osman.jpg")



base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
base_model.summary()
# Maximize the activations of these layers
names = ['mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# Create the feature extraction model
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)



deeptransfer = DeepTransfer(dream_model)

def run_deep_transfer_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    print(img.shape)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    print(img.shape)
    img = tf.convert_to_tensor(img)
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

        # loss, img = deepdream(img, run_steps, tf.constant(step_size))
        loss, img = deeptransfer(img, run_steps, step_size)


        plt.imshow(deprocess(img))
        plt.draw()
        plt.cla()
        print ("Step {}, loss {}".format(step, loss))


    result = deprocess(img)
    plt.imshow(result)
    plt.show()

    return result

new_img = run_deep_transfer_simple(img=original_img, steps=100, step_size=0.01)
