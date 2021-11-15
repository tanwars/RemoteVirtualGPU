import tensorflow as tf
import numpy as np

import time

## here batch images is np array of size (batch, 224, 224, 3)
## each image is normalized between 0 and 1

# with tf.device("GPU:0"):
# tf.debugging.set_log_device_placement(True)
model = tf.keras.models.load_model('model.h5')

print("Here ---------------------------------")

# logits = model(batch_images)  
# prediction = np.argmax(logits, axis=1)
# images_tf = tf.random.uniform(shape = [64, 224, 224, 3])

images_np = np.random.randn(64, 224, 224, 3)
images_tf = tf.convert_to_tensor(images_np, dtype = tf.float32)

ts = time.time()
# logits = model.predict(images_tf, batch_size = 64)
logits = model(images_tf)  
te = time.time()

print('inference time taken: ', te-ts)

# # print(model.device)
# # print(images_tf.device)


# prediction = np.argmax(logits, axis=1)

# ## send predictions

print(model.summary())
# print(images_tf.device)
