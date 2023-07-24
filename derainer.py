import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import load_img, img_to_array
from CVAE import *
import matplotlib.pyplot as plt


cvae = VAE(encoder, decoder)

dummy_input = np.random.rand(1, *img_shape).astype('float32')
cvae(dummy_input)

cvae.load_weights('./my_model.h5')


# remove raindrops from images using trained model
def derain(image):
    # preprocessing
    image = img_to_array(image)
    image = image / 255.
    image = np.expand_dims(image, axis=0)

    derained_image = cvae.predict(image)

    return derained_image

sample = load_img('train/data/0_rain.png', target_size=img_shape)
label = load_img('train/gt/0_clean.png', target_size=img_shape)


derained_image = derain(sample)[0]

plt.imsave('sample_derained.png', derained_image)