
python -m baseline.client --config config/confidence/vit.json \
    --model_path /mnt/xly/checkpoints/vit-tiny-patch16-224 \
    --dataset_path /mnt/xly/ImageNet \
    --server_type triton \
    --url localhost:8001
