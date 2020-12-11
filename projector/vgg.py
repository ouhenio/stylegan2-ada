import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.losses import CosineSimilarity
import keras.backend as K

class PerceptualModel:
    def __init__(self, img_size, layer=-2, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size

        self.perceptual_model = None
        self.ref_img_features = None
        self.features_weight = None
        self.loss = None

    def get_output_for(self, original_image, generated_image):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        self.perceptual_model = Model(vgg16.input, vgg16.layers[self.layer].output)
        # generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
        #                                                           (self.img_size, self.img_size), method=1))
        generated_image_features = self.perceptual_model(generated_image)
        original_image_features = self.perceptual_model(original_image)

        cosine_loss = CosineSimilarity()
        return -cosine_loss(original_image_features, generated_image_features)