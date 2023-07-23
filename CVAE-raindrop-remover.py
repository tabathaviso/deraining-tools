import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def load_dataset(data_dir_rainy, data_dir_clean, img_size=(720, 480)):
    rainy_images = []
    clean_images = []

    rainy_image_files = os.listdir(os.path.join(data_dir_rainy, 'rainy'))
    clean_image_files = os.listdir(os.path.join(data_dir_clean, 'clean'))

    for filename in rainy_image_files:
        if filename.endswith("_rain.png"):
            rainy_img_path = os.path.join(data_dir, 'rainy', filename)
            rainy_img = cv2.imread(rainy_img_path)
            rainy_img = cv2.resize(rainy_img, img_size)
            rainy_images.append(rainy_img)

    for filename in clean_image_files:
        if filename.endswith("_clean.png"):
            clean_img_path = os.path.join(data_dir, 'clean', filename)
            clean_img = cv2.imread(clean_img_path)
            clean_img = cv2.resize(clean_img, img_size)
            clean_images.append(clean_img)

    rainy_images = np.array(rainy_images)
    clean_images = np.array(clean_images)

    return rainy_images, clean_images

# import
data_dir_rainy = '/Users/tviso/Desktop/Github/deraining-tools/rainy'
data_dir_clean = '/Users/tviso/Desktop/Github/deraining-tools/clean'
rainy_images, clean_images = load_dataset(data_dir_rainy, data_dir_clean)
input_shape = (720, 480, 3)

# Visualize the first few images from the dataset
num_images_to_show = 5

fig, axs = plt.subplots(2, num_images_to_show, figsize=(15, 6))

for i in range(num_images_to_show):
    axs[0, i].imshow(rainy_images[i])
    axs[0, i].axis('off')
    axs[0, i].set_title(f"Rainy Image {i}")

    axs[1, i].imshow(clean_images[i])
    axs[1, i].axis('off')
    axs[1, i].set_title(f"Clean Image {i}")

plt.show()
#
# # preprocessing, normalize pixel values to [0, 1]
# rainy_images = rainy_images.astype('float32') / 255.0
# clean_images = clean_images.astype('float32') / 255.0
#
# # CVAE architecture
# latent_dim = 128
#
# encoder = tf.keras.Sequential([
#     # input shape for custom images
#     tf.keras.layers.InputLayer(input_shape=input_shape),
#
#     # convolutional layers, ReLU activation function, 2x2 stride for down-sampling, zero-padding
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'),
#
#     # flatten and output for mean and log-var
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(2 * latent_dim),
# ])
#
# decoder = tf.keras.Sequential([
#     # input shape for latent variable, reverse encoder flattening, reshape
#     tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
#     tf.keras.layers.Dense(90 * 60 * 64, activation='relu'),
#     tf.keras.layers.Reshape((90, 60, 64)),
#
#     # de-convolutional layers, ReLU, 2x2 stride to increase spatial dimensions of output map, zero-padding
#     tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same'),
#     tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
#     tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same'),
#     # outputs reconstructed color image
# ])
#
#
# # re-parameterization trick
# class Sampling(tf.keras.layers.Layer):
#     def call(self, inputs):
#         mean, log_var = inputs  # inputs passed from encoder
#         epsilon = tf.random.normal(shape=tf.shape(mean))    # random noise sampled from std Gaussian distribution
#         # returns a sample from latent variable distribution to maintain stochasticity + allow backpropagation
#         return mean + tf.exp(0.5 * log_var) * epsilon
#
#
# # connect encoder + decoder with sampling layer, create CVAE model
# latent_variable = Sampling()([encoder.output[:, :latent_dim], encoder.output[:, latent_dim:]])
# cvae = tf.keras.Model(inputs=encoder.input, outputs=decoder(latent_variable))
#
#
# # loss functions
# def reconstruction_loss(x, x_decoded_mean):
#     return tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
#
# def kl_loss(mean, log_var):
#     return -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)
#
# def cvae_loss(x, x_decoded_mean, mean, log_var):
#     reconstruction_loss_val = reconstruction_loss(x, x_decoded_mean)
#     kl_loss_val = kl_loss(mean, log_var)
#     # total loss which model aims to minimize
#     return tf.reduce_mean(reconstruction_loss_val + kl_loss_val)
#
# # optimization method
# optimizer = tf.keras.optimizers.Adam()
#
# # compile + train
# cvae.compile(optimizer=optimizer, loss=cvae_loss)
# batch_size = 64
# epochs = 20
# history = cvae.fit(rainy_images, clean_images, batch_size=batch_size, epochs=epochs)
#
# # remove raindrops from images using trained model
# def derain(image):
#     # preprocessing
#     image = image.astype('float32') / 255.0
#     image = np.expand_dims(image, axis=0)
#
#     # encode
#     mean, log_var = encoder.predict(image)
#     latent_variable = np.random.normal(mean, np.exp(0.5 * log_var))
#
#     # decode
#     derained_image = decoder.predict(latent_variable)
#
#     # return derained image
#     derained_image = np.squeeze(derained_image)
#     derained_image = np.clip(derained_image, 0.0, 1.0)
#     derained_image = (derained_image * 255).astype('uint8')
#     return derained_image
