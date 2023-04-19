MODEL=$1
VERBOSE=$2

python -m scripts.convert_hf_ckpt_to_torchscript \
    --model_name google/${MODEL} \
    --model_path /mnt/raid0nvme1/xly/.cache/ \
    --num_gpu 1 --cfg_only

python -m scripts.gen_t5x_agg_cfg --model_name ${MODEL}
python -m scripts.gen_t5x_ensemble_cfg --model_name ${MODEL}

docker stop tritonserver_custom
docker rm tritonserver_custom
docker run --gpus='"device=3"' \
    --name tritonserver_custom \
    --shm-size=30gb \
    --cpuset-cpus=0-27 \
    -d --net=host \
    -v ${PWD}/model_repo_${MODEL}:/models \
    -v ${HOME}/core/build/install/lib:/opt/tritonserver/lib \
    -v ${HOME}/pytorch_backend/build/install/backends/pytorch:/opt/tritonserver/backends/pytorch \
    -v ${HOME}/muduo/build/lib:/root/lib \
    tritonserver_custom:dev \
    tritonserver --model-repository=/models \
        --exit-on-error true \
        --allow-metrics true \
        --http-port 60050 \
        --grpc-port 60051 \
        --metrics-port 60052 \
        --model-control-mode explicit \
        --load-model=* \
        --log-verbose ${VERBOSE}

# docker run --gpus='"device=3,4"' \
#     --name tritonserver_custom \
#     --shm-size=30gb \
#     -it --rm --net=host \
#     -v ${PWD}/model_repo_switch-base-8:/models \
#     -v ${HOME}/core/build/install/lib:/opt/tritonserver/lib \
#     -v ${HOME}/pytorch_backend/build/install/backends/pytorch:/opt/tritonserver/backends/pytorch \
#     -v ${HOME}/muduo/build/lib:/root/lib \
#     tritonserver_custom:dev bash

# gdb --args tritonserver --model-repository=/models \
#     --exit-on-error true \
#     --allow-metrics true \
#     --model-control-mode explicit \
#     --load-model=* \
#     --http-port 50050 \
#     --grpc-port 50051 \
#     --metrics-port 50052 \
#     --log-verbose 0

# docker cp ${HOME}/pytorch_backend/build/install/backends triton_custom:/opt/tritonserver/backends/
# 
# docker run --gpus=1 \
#     --rm -d --net=host \
#     -v ${PWD}/model_repository:/models \
#     nvcr.io/nvidia/tritonserver:22.08-py3 \
#     tritonserver --model-repository=/models \
#         --exit-on-error false \
#         --model-control-mode explicit \
#         --load-model=* 

# nvcr.io/nvidia/tritonserver:22.08-py3 \
