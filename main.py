import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model



"""
This should be where we do the whole algorithm

TODO: figure out how to get gradient for image I at layer x
(started a little at the bottom but i think it is the wrong tf version)


"""

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE, weights='imagenet')


base_model.trainable = False
base_model.summary()


# specify layers we want
layer_name = "conv2_block1_0_bn"
layer_output = base_model.get_layer(layer_name).output

# make new modelwhich outputs these layers
model = Model(base_model.input, outputs=[layer_output])


# eval
input_data = np.random.rand(1, *IMG_SHAPE)
result = model.predict(input_data)
print(result[0].shape)


def get_grad_for_image(session, grad, img):
    #TODO might need to run on subsets of image
    feed_dict = model.create_feed_dict(image=img)
    g = session.run(gradient, feed_dict=feed_dict)

    #TODO normalize
    return g



session = tf.compat.v1.Session(graph=model.graph)
layer_tensor = model.layer_tensors[2]
print(layer_tensor)
grad = model.get_gradient(layer_tensor)
