python -m baseline.client --config config/confidence/vit.json \
    --model_path /mnt/xly/checkpoints/vit-tiny-patch16-224 \
    --dataset_path /mnt/xly/ImageNet \
    --server_type deepspeed \
    --verbose \
    --url localhost:50051