docker run --gpus=1 \
    --name triton_custom \
    -d --net=host \
    -v ${PWD}/model_repository:/models \
    -v ${HOME}/pytorch_backend/build/install/backends:/opt/tritonserver/backends/ \
    tritonserver_custom \
    tritonserver --model-repository=/models \
        --exit-on-error false \
        --model-control-mode explicit \
        --load-model=* 

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