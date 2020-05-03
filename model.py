import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from laplacian import compute_laplacian


class DeepTransfer(tf.Module):
    def __init__(self, dream_model, style_model):
        self.dream_model = dream_model
        self.style_model = style_model

    # @tf.function(
    #     input_signature=(
    #         tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
    #         tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
    #         tf.TensorSpec(shape=[], dtype=tf.int32),
    #         tf.TensorSpec(shape=[], dtype=tf.float32),)
    # )
    def __call__(self, source, target, steps, step_size):

        source_style = self.style(source)
        original = tf.identity(target)

        M = compute_laplacian(source.numpy())

        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(target)
                loss = self.loss(source_style, target, original)

            gradients = tape.gradient(loss, target)

            gradients /= tf.math.reduce_std(gradients) + 1e-8
            
            photoreal_weight = 0.5
            photo_grad = -(2 * M.dot(tf.reshape(target, (-1,3)))).reshape(target.shape) * photoreal_weight
            photo_grad[photo_grad < gradients] = 0

            target = target + gradients*step_size
<<<<<<< HEAD
            #target = tf.clip_by_value(target, -1, 1)
=======
            target = target + photo_grad*step_size
            target = tf.clip_by_value(target, -1, 1)
>>>>>>> 924aa1f8d298bc03e06afb40fd7d0c0791fdc68b

        return loss, target

    def gram_matrix(self, tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
        input_shape = tf.shape(tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def style(self, img):
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = self.style_model(img_batch)
        if not type(layer_activations) is list:
          layer_activations = [layer_activations]

        img_style = [self.gram_matrix(activation) for activation in layer_activations]
        return img_style

    def dream_loss(self, target):
        img_batch = tf.expand_dims(target, axis=0)
        layer_activations = self.dream_model(img_batch)
        if len(layer_activations) == 1:
          layer_activations = [layer_activations]

        losses = []
        for act in layer_activations:
          loss = tf.math.reduce_mean(act)
          losses.append(loss)

        return  tf.reduce_sum(losses)

    def style_loss(self, source_style, target):
        num_style_layers = len(source_style)
        target_style = self.style(target)
        style_loss = tf.add_n([tf.reduce_mean((source_style[i]-target_style[i])**2)
                           for i in range(num_style_layers)])
        style_loss /= num_style_layers
        return style_loss

    def content_loss(self, original, target):
        img_batch = tf.expand_dims(original, axis=0)
        orig_act = self.dream_model(img_batch)
        if not type(orig_act) is list:
            orig_act = [orig_act]

        img_batch = tf.expand_dims(target, axis=0)
        targ_act = self.dream_model(img_batch)
        if not type(targ_act) is list:
            targ_act = [targ_act]

        content_loss = tf.add_n([tf.reduce_mean(tf.square(orig_act[i] - targ_act[i])) for i in range(len(orig_act))])
        return content_loss / len(orig_act)


    def loss(self, source_style, target, original):
        style_weight = 1
<<<<<<< HEAD
        return - style_weight * self.style_loss(source_style, target) #+  self.dream_loss(target)
=======
        return -style_weight * self.style_loss(source_style, target) - self.content_loss(original, target) #+ self.dream_loss(target)
>>>>>>> 924aa1f8d298bc03e06afb40fd7d0c0791fdc68b
