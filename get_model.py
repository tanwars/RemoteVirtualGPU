import tensorflow as tf

print(tf.__version__)


IMAGE_SIZE = 224

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

base_model.save('mobilenetV2.h5')
