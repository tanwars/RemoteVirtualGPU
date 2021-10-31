import tensorflow as tf
import numpy as np

## here batch images is np array of size (batch, 224, 224, 3)
## each image is normalized between 0 and 1

model = tf.keras.models.load_model('model.h5')

# logits = model(batch_images)  
# prediction = np.argmax(logits, axis=1)

## send predictions

print(model.summary())
