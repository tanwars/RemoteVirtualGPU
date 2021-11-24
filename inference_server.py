from concurrent import futures
import logging

import grpc
import inferencedata_pb2
import inferencedata_pb2_grpc

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

from PIL import Image
import io

import time

class RemoteInference(inferencedata_pb2_grpc.RemoteInferenceServicer):

    def __init__(self, *args, **kwargs):
        # load the model
        try:
            # tf.debugging.set_log_device_placement(True)
            self.model = tf.keras.models.load_model(kwargs['model'], custom_objects={'KerasLayer': hub.KerasLayer})
            images_np = np.random.randn(64, 224, 224, 3)
            self.model.predict(images_np, batch_size=64)
            print('server is running')
        except Exception as e:
            print('Model not loaded properly')
            print(e)

    def Infer(self, request, context):
        resultbatch = inferencedata_pb2.ResultBatch()
        input_batch_size = len(request.images)

        # TODO: may want to create custom batch sizing here based on GPU and network

        # TODO: makes sure image is of the right size (assumes 224,224,3)
        images_np = np.empty((input_batch_size, 224, 224, 3)) 

        for i, image in enumerate(request.images):
            # convert image to np array
            image_PIL = Image.open(io.BytesIO(image.image_data))
            # image_PIL = image_PIL.resize((224,224))
            image_np = np.array(image_PIL)

            # TODO: makes sure image is of the right size (assumes 224,224,3)
            assert image_np.shape[0]==224, 'Expects dim 224'
            assert image_np.shape[1]==224, 'Expects dim 224'
            assert image_np.shape[2]==3, 'Expects dim 3' 
            
            # append to an array to create complete batch
            images_np[i,:,:,:] = image_np
        
        # pass it to model to process
        images_np = images_np * 1./255 ## rescales it for the model

        # with tf.device("GPU:0"):
        #     images_tf = tf.convert_to_tensor(images_np, dtype = tf.float32)

        ts = time.time()
        logits = self.model.predict(images_np, batch_size = 64)  
        te = time.time()
        
        prediction = np.argmax(logits, axis=1)

        # convert back to result type
        for i in range(input_batch_size):
            result = resultbatch.results.add()
            result.id = i
            result.num = prediction[i]

        print('time taken:', te-ts)

        return resultbatch

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inferencedata_pb2_grpc.add_RemoteInferenceServicer_to_server(RemoteInference(model = 'models/resnet_v2_1.0_224.h5'), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
