python -m baseline.client --config config/confidence/vit.json \
    --model_path /mnt/xly/checkpoints/vit-tiny-patch16-224 \
    --dataset_path /mnt/xly/ImageNet \
    --server_type deepspeed \
    --verbose \
    --url localhost:50051

# docker run --gpus=1 --rm -d --net=host -v ${PWD}/baseline/model_repository:/models nvcr.io/nvidia/tritonserver:22.08-py3 tritonserver --model-repository=/models --model-control-mode poll --repository-poll-secs 5
# sleep 5
# python -m baseline.client --config config/confidence/vit.json \
#     --model_path /mnt/xly/checkpoints/vit-tiny-patch16-224 \
#     --dataset_path /mnt/xly/ImageNet \
#     --server_type triton \
#     --url localhost:8001