import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_dataset():
    import glob
    from tqdm import tqdm

    input_glob = sorted(glob.glob('/home/nikesh/Downloads/CS539/train/data/*.png'))
    ground_glob = sorted(glob.glob('/home/nikesh/Downloads/CS539/train/gt/*.png'))

    input_images = []
    ground_truth = []

    for i in tqdm(input_glob):
        img = tf.keras.utils.load_img(i, target_size=(120, 180, 3))
        img = tf.keras.utils.img_to_array(img)
        img = img / 255.
        input_images.append(img)

    for j in tqdm(ground_glob):
        img = tf.keras.utils.load_img(j, target_size=(120, 180, 3))
        img = tf.keras.utils.img_to_array(img)
        img = img / 255.
        ground_truth.append(img)

    input_images = np.array(input_images)
    ground_truth = np.array(ground_truth)

    return input_images, ground_truth


# import
rainy_images, clean_images = load_dataset()
input_shape = (120, 180, 3)



# preprocessing, normalize pixel values to [0, 1]
# rainy_images = rainy_images.astype('float32') / 255.0
# clean_images = clean_images.astype('float32') / 255.0

# CVAE architecture
latent_dim = 32

encoder = tf.keras.Sequential([
    # input shape for custom images
    tf.keras.layers.InputLayer(input_shape=input_shape),

    # convolutional layers, ReLU activation function, 2x2 stride for down-sampling, zero-padding
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same'),

    # flatten and output for mean and log-var
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),  # Adjusted layer to match decoder's input shape
    tf.keras.layers.Dense(2 * latent_dim),
])
#
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
decoder_input = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(90 * 60 * 64, activation='relu')(decoder_input)
x = tf.keras.layers.Reshape((90, 60, 64))(x)
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
decoder_output = tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)

encoder.summary()
decoder.summary()


# re-parameterization trick
class Sampling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.run_eagerly = True

    def call(self, inputs):
        mean, log_var = inputs  # inputs passed from encoder
        epsilon = tf.random.normal(shape=tf.shape(mean))  # random noise sampled from std Gaussian distribution
        # returns a sample from latent variable distribution to maintain stochasticity + allow backpropagation
        return mean + tf.exp(0.5 * log_var) * epsilon


# connect encoder + decoder with sampling layer, create CVAE model
latent_variable = Sampling()([encoder.output[:, :latent_dim], encoder.output[:, latent_dim:]])
cvae = tf.keras.Model(inputs=encoder.input, outputs=decoder(latent_variable))


# loss functions
def reconstruction_loss(x, x_decoded_mean):
    return tf.keras.losses.binary_crossentropy(x, x_decoded_mean)


def kl_loss(mean, log_var):
    return -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=-1)


def cvae_loss(x, x_decoded_mean, mean, log_var):
    reconstruction_loss_val = reconstruction_loss(x, x_decoded_mean)
    kl_loss_val = kl_loss(mean, log_var)
    # total loss which model aims to minimize
    return tf.reduce_mean(reconstruction_loss_val + kl_loss_val)


# optimization method
optimizer = tf.keras.optimizers.Adam()

encoder.summary()
decoder.summary()
# compile + train
cvae.compile(optimizer=optimizer, loss=cvae_loss, run_eagerly=True)
batch_size = 10
epochs = 20
history = cvae.fit(rainy_images, clean_images, batch_size=batch_size, epochs=epochs)


# remove raindrops from images using trained model
def derain(image):
    # preprocessing
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # encode
    mean, log_var = encoder.predict(image)
    latent_variable = np.random.normal(mean, np.exp(0.5 * log_var))

    # decode
    derained_image = decoder.predict(latent_variable)

    # return derained image
    derained_image = np.squeeze(derained_image)
    derained_image = np.clip(derained_image, 0.0, 1.0)
    derained_image = (derained_image * 255).astype('uint8')
    return derained_image
