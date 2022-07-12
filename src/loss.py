import tensorflow as tf
import numpy as np

def wasserstein_generator_loss(real_img, fake_img, **kwargs):
    
    return tf.keras.backend.mean(fake_img)

def wasserstein_discriminator_loss(real_img, fake_img, **kwargs):
    
    return tf.keras.backend.mean(fake_img)
    # return tf.keras.backend.mean(fake_img) - tf.keras.backend.mean(real_img)
    

# https://stackoverflow.com/questions/59039886/get-randomly-weighted-averages-between-samples-in-a-batch-with-arbitrary-sample
# https://stackoverflow.com/questions/58133430/how-to-substitute-keras-layers-merge-merge-in-tensorflow-keras
class RandomWeightedAverage(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.batch_size = 16 # TODO hardcoded FIX THIS

    def call(self, inputs, **kwargs):
        alpha = tf.keras.backend.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

# https://stackoverflow.com/questions/61058119/implementing-gradient-penalty-loss-with-tensorflow-2
class GradientPenaltyLoss(tf.keras.losses.Loss):
    
    def __init__(self, averaged_imgs, gradient_penalty_weight=10):
        
        super().__init__()
        self.averaged_imgs = averaged_imgs
        self.gradient_penalty_weight = gradient_penalty_weight
        
    def call(self, real_img, fake_img, **kwargs):
        
        gradients = tf.keras.backend.gradients(fake_img, self.averaged_imgs)[0]
        gradients_sqr = tf.keras.backend.square(gradients)
        gradients_sqr_sum = tf.keras.backend.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = tf.keras.backend.sqrt(gradients_sqr_sum)
        gradient_penalty = self.gradient_penalty_weight * tf.keras.backend.square(1 - gradient_l2_norm)
        return tf.keras.backend.mean(gradient_penalty)