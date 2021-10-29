from concurrent import futures
import logging

import grpc
import inferencedata_pb2
import inferencedata_pb2_grpc

class RemoteInference(inferencedata_pb2_grpc.RemoteInferenceServicer):

    def Infer(self, request, context):
        resultbatch = inferencedata_pb2.ResultBatch()
        for i, image in enumerate(request.images):
            result = resultbatch.results.add()
            result.id = image.id
            result.num = i + 1
        return resultbatch

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inferencedata_pb2_grpc.add_RemoteInferenceServicer_to_server(RemoteInference(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()