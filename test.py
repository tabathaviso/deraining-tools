import tensorflow as tf
from keras import Model, Input, regularizers
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.callbacks import EarlyStopping
import numpy as np

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
def load_dataset():
    import glob
    from tqdm import tqdm

    input_glob = sorted(glob.glob('/home/nikesh/Downloads/CS539/train/train/data/*.png'))
    ground_glob = sorted(glob.glob('/home/nikesh/Downloads/CS539/train/train/data/*.png'))

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

Input_img = Input(shape=(120, 180, 3))

# encoding architecture
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x2 = MaxPool2D((2, 2))(x2)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)

# decoding architecture
x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x3 = UpSampling2D((2, 2))(x3)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
decoded = Conv2D(3, (3, 3), padding='same')(x1)

autoencoder = Model(Input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', run_eagerly=True)

autoencoder.summary()

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1, mode='auto')

a_e = autoencoder.fit(rainy_images, clean_images,
                      epochs=50,
                      batch_size=10,
                      shuffle=True,
                      callbacks=[early_stopper])
