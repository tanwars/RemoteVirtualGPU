# RemoteVirtualGPU

### Test the protobuf snippet

1. [optional] Recompile the Protocol Buffers: `python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./inferencedata.proto`

2. Start the server: `python3 inference_server.py`

3. Run the client: `python3 inference_client.py`
