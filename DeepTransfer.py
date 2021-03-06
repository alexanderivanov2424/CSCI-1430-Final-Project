import tensorflow as tf
import argparse 
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from model import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str,
    default='Picasso',
    help='Filename of the source image (example: Picasso)')

    parser.add_argument('--target', type=str,
    default='Osman',
    help='Filename of the target image (example: Osman)')

    parser.add_argument('--size', type=int,
    default=200,
    help='Size in pixels of the output image (default: 200px)')

    args = parser.parse_args()
    return args

def main():
  global args
  args = parse_args()

if __name__ == '__main__':
  main()

# Download an image and read it into a NumPy array.
def get_image_as_array(file_name, size=args.size):
    img = Image.open(file_name)
    img = img.resize((size,size))
    img = img.convert('RGB')
    return np.array(img)

source = get_image_as_array("./RawImages/"+args.source+".jpg")
target = get_image_as_array("./RawImages/"+args.target+".jpg")
plt.imshow(target)
plt.show()


#base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
base_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
# RESNET base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
base_model.summary()

# Maximize the activations of these layers
dream_names = ['block1_pool']

# Style is a function of these layers' activations
style_names = ['block1_conv1',
                'block2_conv1',
                 'block3_conv1',
                 'block4_conv1',
                 'block5_conv1']
#resnet style_names = ['conv4_block1_1_conv']
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


        plt.imshow(img + .0)
        plt.show()
        print ("Step {}, loss {}".format(step, loss))


    #plt.imshow(img)
    #plt.show()

    return img

new_img = run_deep_transfer(source, target, steps=50, step_size=0.01)

