from __future__ import print_function

import logging

import grpc
import inferencedata_pb2
import inferencedata_pb2_grpc

# image manipulation specific libraries
from PIL import Image
import numpy as np
import io

import time

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('ec2-54-215-143-22.us-west-1.compute.amazonaws.com:50052') as channel:
        img = Image.open('marguerite-729510__480.jpg')
        img = img.resize((224, 224))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        imagebatch = inferencedata_pb2.ImageBatch()
        for i in range(1):
            imagepb = imagebatch.images.add()
            imagepb.id = i + 1
            imagepb.image_data = img_byte_arr

        stub = inferencedata_pb2_grpc.RemoteInferenceStub(channel)

        i = 0
        total = 0.0
        while True:
            st = time.perf_counter()
            response = stub.Infer(imagebatch)
            total += ((time.perf_counter() - st) * 1000)
            i += 1
            if i % 10 == 0:
                print(i, total / i)
    print("Inference client received: {}".format(response))

if __name__ == '__main__':
    logging.basicConfig()
    run() 
