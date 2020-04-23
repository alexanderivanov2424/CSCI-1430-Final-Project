import tensorflow as tf
import numpy as np


class DeepTransfer(tf.Module):
    def __init__(self, model, source_image):
        self.model = model
        self.source_activations = model.predict(tf.expand_dims(source_image, axis=0))

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)
                loss = calc_loss(img, self.model)

            gradients = tape.gradient(loss, img)

            gradients /= tf.math.reduce_std(gradients) + 1e-8

            img = img + gradients*step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img

    def loss(img, model):
        # Pass forward the image through the model to retrieve the activations.
        # Converts the image into a batch of size 1.
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)
        if len(layer_activations) == 1:
          layer_activations = [layer_activations]

        losses = []
        for act in layer_activations:
          loss = tf.math.reduce_mean(act)
          losses.append(loss)

        return  tf.reduce_sum(losses)
