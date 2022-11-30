python -m pysrc.client --config config/confidence/vit.json \
    --model_path /mnt/xly/checkpoints/vit-tiny-patch16-224 \
    --dataset_path /mnt/xly/ImageNet \
    --server_type deepspeed \
    --verbose \
    --url localhost:50051


 python -m pysrc.single_vit_client \
    --dataset_path /mnt/xly/ImageNet \
    --server_type deepspeed \
    --model_name vit-xl-patch16-224 \
    --model_path /mnt/xly/checkpoints/vit-tiny-patch16-224 \
    --config config/confidence/vit.json \
    --url localhost:50051