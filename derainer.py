import numpy as np
import tensorflow as tf
from CVAE import *

# Change this variable to wherever your test dataset is located. 
# This follows the same format as the training dataset.
TEST_DATASET_PATH = 'test_a'

cvae = CVAE(encoder, decoder)

# random input to ensure model is loaded
dummy_input = np.random.rand(1, *img_shape).astype('float32')
cvae(dummy_input)

# loading pretrained model
cvae.load_weights('./good_model.h5')

# loading test dataset
test_rain, _ = load_dataset(TEST_DATASET_PATH)


#batch deraining

derained_images = cvae(test_rain)

# Denormalizing image data
derained_images = derained_images * 255
derained_images = tf.cast(derained_images, tf.uint8)

for i, output_img in enumerate(derained_images):
    # Encode tensor to PNG
    encoded_img = tf.image.encode_png(output_img)

    # Write to file
    tf.io.write_file(f'{TEST_DATASET_PATH}/prediction/{i}_prediction.png', encoded_img)

