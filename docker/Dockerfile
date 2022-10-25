FROM nvcr.io/nvidia/tritonserver:22.08-py3

# Copy the custom onnxruntime to the container
COPY onnxruntime_backend/backends/onnxruntime/libtriton_onnxruntime.so /opt/tritonserver/backends/onnxruntime/libtriton_onnxruntime.so
COPY pytorch_backend/backends/pytorch/libtriton_pytorch.so /opt/tritonserver/backends/pytorch/libtriton_pytorch.so