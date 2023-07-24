import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import load_img, img_to_array
from keras.callbacks import LearningRateScheduler

img_shape = (240, 360, 3)

def load_dataset():
    import glob
    from tqdm import tqdm
    
    input_glob = sorted(glob.glob('train/data/*.png'))
    ground_glob = sorted(glob.glob('train/gt/*.png'))

    input_images = []
    ground_truth = []

    for i in tqdm(input_glob):
        img = load_img(i, target_size=img_shape)
        img = img_to_array(img)
        img = img / 255.
        input_images.append(img)

    for j in tqdm(ground_glob):
        img = load_img(j, target_size=img_shape)
        img = img_to_array(img)
        img = img / 255.
        ground_truth.append(img)

    input_images = np.array(input_images)
    ground_truth = np.array(ground_truth)

    return input_images, ground_truth

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# CVAE architecture
latent_dim = 64

encoder_inputs = keras.Input(shape=img_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(encoder_inputs)
x = layers.Conv2D(64, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
# encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(15 * 23 * 256, activation='relu')(latent_inputs)
x = layers.Reshape((15, 23, 256))(x)
x = layers.Conv2DTranspose(128, (1, 1), activation='relu', strides=(1, 1), padding='valid')(x)
x = layers.Conv2DTranspose(128, (4, 4), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2DTranspose(64, (4, 4), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2DTranspose(64, (4, 4), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2DTranspose(32, (4, 4), activation='relu', strides=(2, 2), padding='same')(x)
x = layers.Conv2DTranspose(32, (4, 4), activation='relu', strides=(2, 2), padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            data, labels = inputs[0]
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(labels, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, input, training=False):
        z_mean, z_log_var, z = self.encoder(input)
        reconstruction = self.decoder(z)
        return reconstruction

def learning_rate_schedule(epoch):
    initial_lr = 0.001  # initial
    if epoch < 10:
        return initial_lr
    else:
        return initial_lr * tf.math.exp(0.1 * (10 - epoch))  # reduce learning rate exponentially after 10 epochs

if __name__ == '__main__':
    cvae = VAE(encoder, decoder)

    rainy_images, clean_images = load_dataset()

    # optimization method
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule(0)) 

    # compile + train
    cvae.compile(optimizer=optimizer, run_eagerly=True)
    lr_scheduler = LearningRateScheduler(learning_rate_schedule)
    batch_size = 5
    epochs = 30
    history = cvae.fit((rainy_images, clean_images), batch_size=batch_size, epochs=epochs, callbacks=[lr_scheduler])

    cvae.save_weights('my_model.h5')





