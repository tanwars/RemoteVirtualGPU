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
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('') as channel:
        files = os.listdir('flower_photos/roses')[:1]
        print(len(files))
        imagebatch = inferencedata_pb2.ImageBatch()

        st = time.time()
        for i, f in enumerate(files):

            with open(os.path.join('flower_photos/roses/', f), 'rb') as fp:

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

        st = time.time()
        stub = inferencedata_pb2_grpc.RemoteInferenceStub(channel)
        response = stub.Infer(imagebatch)
        en = time.time()

        print("Took {} seconds to receive the results".format(en - st))

    # print("Inference client received {}".format(response))
    print("Inference client received response of length {} and first item {}".format(len(response.results), response.results[0]))

if __name__ == '__main__':
    logging.basicConfig()
    run()

