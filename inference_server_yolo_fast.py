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

# import yolo
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array

from yolo_keras.yolo import YOLO

import argparse

PARAM_IMG_SIZE = 300
PARAM_INPUT_SIZE = 416

class RemoteInference(inferencedata_pb2_grpc.RemoteInferenceServicer):

    def __init__(self, *args, **kwargs):
        # load the model
        try:
            # tf.debugging.set_log_device_placement(True)
            # self.model = tf.keras.models.load_model(kwargs['model'], custom_objects={'KerasLayer': hub.KerasLayer})
            # images_np = np.random.randn(64, PARAM_INPUT_SIZE, PARAM_INPUT_SIZE, 3)
            # print(self.model.summary())
            # print('setting up')
            # self.model.predict(images_np, batch_size=64)
            # print('server is running')
            self.yolo = YOLO(**vars(kwargs['flags']))
            # images_np = np.random.randn(1, PARAM_INPUT_SIZE, PARAM_INPUT_SIZE, 3)
            # r_image = yolo.detect_image(images_np)
            print('server is running')

        except Exception as e:
            print('Model not loaded properly')
            print(e)

    def Infer(self, request, context):

        ts = time.perf_counter()

        resultbatch = inferencedata_pb2.ResultBatch()
        input_batch_size = len(request.images)

        assert input_batch_size == 1, "currently only supports one image at a time"
        # TODO: may want to create custom batch sizing here based on GPU and network

        # TODO: makes sure image is of the right size (assumes 224,224,3)
        # images_np = np.empty((input_batch_size, PARAM_INPUT_SIZE, PARAM_INPUT_SIZE, 3)) 

        for i, image in enumerate(request.images):
            # image_PIL = load_img(io.BytesIO(image.image_data), target_size=(PARAM_INPUT_SIZE, PARAM_INPUT_SIZE))
            # image_PIL = img_to_array(image_PIL)
            # convert image to np array
            image_PIL = Image.open(io.BytesIO(image.image_data))
            # image_PIL = image_PIL.resize((PARAM_INPUT_SIZE,PARAM_INPUT_SIZE))
            # image_np = np.array(image_PIL)

            # # TODO: makes sure image is of the right size (assumes 224,224,3)
            # assert image_np.shape[0]==224, 'Expects dim 224'
            # assert image_np.shape[1]==224, 'Expects dim 224'
            # assert image_np.shape[2]==3, 'Expects dim 3' 
            
            # # append to an array to create complete batch
            # images_np[i,:,:,:] = image_np
        
        # pass it to model to process
        # images_np = images_np * 1./255 ## rescales it for the model

        # with tf.device("GPU:0"):
        #     images_tf = tf.convert_to_tensor(images_np, dtype = tf.float32)

        # ts = time.perf_counter()
        # logits = self.model.predict(images_np, batch_size = 64)  
        v_boxes, v_scores, v_labels = self.yolo.detect_image_bbox(image_PIL)
        # te = time.perf_counter()

        # convert back to result type
        for i in range(input_batch_size):
            result = resultbatch.results.add()
            result.id = i
            for j in range(v_boxes.shape[0]):
                f_box = result.boxes.add()
                f_box.xmin = v_boxes[j,1]
                f_box.xmax = v_boxes[j,3]
                f_box.ymin = v_boxes[j,0]
                f_box.ymax = v_boxes[j,2]
                f_box.label = v_labels[j]
                f_box.score = v_scores[j]
                f_box.id = j

        te = time.perf_counter()

        print('time taken:', te-ts)

        return resultbatch

def serve(FLAGS):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inferencedata_pb2_grpc.add_RemoteInferenceServicer_to_server(RemoteInference(flags = FLAGS), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    
    assert FLAGS.image == True, "image argument must be true"

    logging.basicConfig()
    serve(FLAGS)
