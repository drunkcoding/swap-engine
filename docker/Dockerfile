FROM nvcr.io/nvidia/tritonserver:22.08-py3

RUN apt update
RUN apt install -y libevent-dev libaio-dev

ENV CUDA_LAUNCH_BLOCKING=1
ENV LD_LIBRARY_PATH=LD_LIBRARY_PATH:/root/lib

RUN pip install tritonclient[all]

RUN apt install -y gdb
COPY nsight-systems-2022.5.1_2022.5.1.82-1_amd64.deb /root/
RUN apt install -y /root/nsight-systems-2022.5.1_2022.5.1.82-1_amd64.deb
RUN rm /root/nsight-systems-2022.5.1_2022.5.1.82-1_amd64.deb

RUN apt-get install -y datacenter-gpu-manager  