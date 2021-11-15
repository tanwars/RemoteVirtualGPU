from __future__ import print_function

import logging

import grpc
import inferencedata_pb2
import inferencedata_pb2_grpc

# image manipulation specific libraries
from PIL import Image
import numpy as np
import io
import os
import time

def run():
    def process_response(call_future):
        print('received result')

    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    channel = grpc.insecure_channel('ec2-54-183-131-10.us-west-1.compute.amazonaws.com:50051')
    for i in range(1):
        files = os.listdir('flower_photos_formatted/roses')[:16]
        print(len(files))
        imagebatch = inferencedata_pb2.ImageBatch()

        st = time.time()
        for i, f in enumerate(files):

            with open(os.path.join('flower_photos_formatted/roses/', f), 'rb') as fp:

                # img = Image.open(os.path.join('flower_photos/roses/', f))
                # img_byte_arr = io.BytesIO()
                # img.save(img_byte_arr, format=img.format)
                # img_byte_arr = img_byte_arr.getvalue()

                imagepb = imagebatch.images.add()
                imagepb.id = i + 1
                # imagepb.image_data = img_byte_arr
                imagepb.image_data = fp.read()

        en = time.time()

        print("Took {} seconds to create the image batch".format(en - st))

        # st = time.time()
        stub = inferencedata_pb2_grpc.RemoteInferenceStub(channel)
        call_future = stub.Infer.future(imagebatch)
        call_future.add_done_callback(process_response)
        # response = stub.Infer(imagebatch)
        # en = time.time()

        print("Took {} seconds to receive the results".format(en - st))

    # print("Inference client received {}".format(response))
    # print("Inference client received response of length {} and first item {}".format(len(response.results), response.results[0]))

if __name__ == '__main__':
    logging.basicConfig()
    run()
    while True:
        time.sleep(3600)

