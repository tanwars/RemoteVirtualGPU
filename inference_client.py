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
import threading

target_fps = 25
batch_size = 1
t_start = time.perf_counter()

lock = threading.Lock()
counter = 0

f_contents = []

def run():
    global counter 

    def process_response(call_future):
        global counter 
        lock.acquire()
        counter += len(call_future.result().results)
        curr_val = counter
        lock.release()
        if curr_val % 10 == 0:
            print('{}: {}'.format(time.perf_counter() - t_start, curr_val))

    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    # channel = grpc.insecure_channel('localhost:50051')
    channel = grpc.insecure_channel('ec2-13-57-32-117.us-west-1.compute.amazonaws.com:50051')
    files = os.listdir('flower_photos_formatted/roses')

    for f in files:
        with open(os.path.join('flower_photos_formatted/roses/', f), 'rb') as fp:
            f_contents.append(fp.read())

    sleep_time = 1. / target_fps

    i = 0
    total = 0
    while True:
        imagebatch = inferencedata_pb2.ImageBatch()

        if i >= len(files):
            total += len(files)
            i = 0

        for j in range(i,min(len(files), i+batch_size)):
            imagepb = imagebatch.images.add()
            imagepb.id = i + 1
            imagepb.image_data = f_contents[j]

            if batch_size > 1:
                time.sleep(sleep_time * (batch_size - 1))

        i += min(batch_size, len(files) - i)

        # print("Took {} seconds to create the image batch".format(en - st))

        # st = time.time()
        stub = inferencedata_pb2_grpc.RemoteInferenceStub(channel)
        call_future = stub.Infer.future(imagebatch)
        call_future.add_done_callback(process_response)

        since = time.perf_counter() - t_start
        if since > 20:
            print("waiting for results")
            lock.acquire()
            num_results = counter
            lock.release()
            print("avg fps sustained: {}".format(counter / since))
            # break

        time.sleep(sleep_time)
        # response = stub.Infer(imagebatch)
        # en = time.time()

        # print("Took {} seconds to receive the results".format(en - st))

        # print("Inference client received {}".format(response))
        # print("Inference client received response of length {} and first item {}".format(len(response.results), response.results[0]))

if __name__ == '__main__':
    logging.basicConfig()
    run()
    while True:
        time.sleep(3600)

