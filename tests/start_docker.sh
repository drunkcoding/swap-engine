docker stop tritonserver_custom
docker rm tritonserver_custom
docker run --privileged --gpus='"device=0"' \
    --name tritonserver_custom \
    --shm-size=10gb \
    -d --net=host \
    -v ${PWD}/model_repo_switch-base-8:/models \
    -v ${HOME}/pytorch_backend/build/install/backends/pytorch:/opt/tritonserver/backends/pytorch \
    -v ${HOME}/core/build/install/lib:/opt/tritonserver/lib \
    -v ${HOME}/muduo/build/lib:/root/lib \
    tritonserver_custom \
    tritonserver --model-repository=/models \
        --exit-on-error true \
        --allow-metrics false \
        --model-control-mode explicit \
        --load-model=* \
        --log-verbose 10

# docker cp ${HOME}/pytorch_backend/build/install/backends triton_custom:/opt/tritonserver/backends/

# docker run --gpus=1 \
#     --rm -d --net=host \
#     -v ${PWD}/model_repository:/models \
#     nvcr.io/nvidia/tritonserver:22.08-py3 \
#     tritonserver --model-repository=/models \
#         --exit-on-error false \
#         --model-control-mode explicit \
#         --load-model=* 

# nvcr.io/nvidia/tritonserver:22.08-py3 \
