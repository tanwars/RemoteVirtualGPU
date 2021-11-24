from tensorflow.keras.models import load_model
import tensorflow as tf

# model = load_model('yolo_v3_model.h5')

# model.summary()

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()

# with open('yolov3.tflite', 'wb') as f:
#   f.write(tflite_model)

############################################

IMAGE_SIZE = 224

def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files('/home/ubuntu/yolo_v3/zebra_dir/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

model = load_model('yolo_v3_model.h5')
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# These set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# And this sets the representative dataset so we can quantize the activations
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

with open('yolov3_quant.tflite', 'wb') as f:
  f.write(tflite_model)