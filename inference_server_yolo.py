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

import yolo
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

PARAM_IMG_SIZE = 300
PARAM_INPUT_SIZE = 416

class RemoteInference(inferencedata_pb2_grpc.RemoteInferenceServicer):

    def __init__(self, *args, **kwargs):
        # load the model
        try:
            # tf.debugging.set_log_device_placement(True)
            self.model = tf.keras.models.load_model(kwargs['model'], custom_objects={'KerasLayer': hub.KerasLayer})
            images_np = np.random.randn(64, PARAM_INPUT_SIZE, PARAM_INPUT_SIZE, 3)
            print(self.model.summary())
            print('setting up')
            self.model.predict(images_np, batch_size=64)
            print('server is running')

            # some setup
            self.anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
            self.class_threshold = 0.6
            self.labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
	"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
	"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
	"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
	"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
	"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        except Exception as e:
            print('Model not loaded properly')
            print(e)

    def Infer(self, request, context):

        resultbatch = inferencedata_pb2.ResultBatch()
        input_batch_size = len(request.images)

        # TODO: may want to create custom batch sizing here based on GPU and network

        # TODO: makes sure image is of the right size (assumes 224,224,3)
        images_np = np.empty((input_batch_size, PARAM_INPUT_SIZE, PARAM_INPUT_SIZE, 3)) 

        for i, image in enumerate(request.images):
            # image_PIL = load_img(io.BytesIO(image.image_data), target_size=(PARAM_INPUT_SIZE, PARAM_INPUT_SIZE))
            # image_PIL = img_to_array(image_PIL)
            # convert image to np array
            image_PIL = Image.open(io.BytesIO(image.image_data))
            image_PIL = image_PIL.resize((PARAM_INPUT_SIZE,PARAM_INPUT_SIZE))
            image_np = np.array(image_PIL)

            # # TODO: makes sure image is of the right size (assumes 224,224,3)
            # assert image_np.shape[0]==224, 'Expects dim 224'
            # assert image_np.shape[1]==224, 'Expects dim 224'
            # assert image_np.shape[2]==3, 'Expects dim 3' 
            
            # # append to an array to create complete batch
            images_np[i,:,:,:] = image_np
        
        # pass it to model to process
        images_np = images_np * 1./255 ## rescales it for the model

        # with tf.device("GPU:0"):
        #     images_tf = tf.convert_to_tensor(images_np, dtype = tf.float32)

        # ts = time.perf_counter()
        logits = self.model.predict(images_np, batch_size = 64)  
        # te = time.perf_counter()
        
        ts = time.perf_counter()
        # prediction = np.argmax(logits, axis=1)
        boxes = list()
        for i in range(len(logits)):
	        # decode the output of the network
	        boxes += yolo.decode_netout(logits[i][0], self.anchors[i], 
                                            self.class_threshold, PARAM_INPUT_SIZE, PARAM_INPUT_SIZE)

        yolo.correct_yolo_boxes(boxes, PARAM_IMG_SIZE, PARAM_IMG_SIZE, PARAM_INPUT_SIZE, PARAM_INPUT_SIZE)
        yolo.do_nms(boxes, 0.5)

        v_boxes, v_labels, v_scores = yolo.get_boxes(boxes, self.labels, self.class_threshold)

        # convert back to result type
        for i in range(input_batch_size):
            result = resultbatch.results.add()
            result.id = i
            for j in range(len(v_boxes)):
                f_box = result.boxes.add()
                f_box.xmin = v_boxes[j].xmin
                f_box.xmax = v_boxes[j].xmax
                f_box.ymin = v_boxes[j].ymin
                f_box.ymax = v_boxes[j].ymax
                # f_box.label = v_labels[j]
                f_box.label = 0
                f_box.score = v_scores[j]
                f_box.id = j

        te = time.perf_counter()

        print('time taken:', te-ts)

        return resultbatch

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inferencedata_pb2_grpc.add_RemoteInferenceServicer_to_server(RemoteInference(model = 'models/yolo_v3_model_variable.h5'), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
