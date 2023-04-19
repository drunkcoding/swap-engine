
python -m baseline.client --config config/confidence/vit.json \
    --model_path /mnt/xly/checkpoints/vit-tiny-patch16-224 \
    --dataset_path /mnt/xly/ImageNet \
    --server_type triton \
    --url localhost:8001


deepspeed --include "localhost:0,1,2,3" tests/python/ds_inference.py \
    --model_name switch-base-128 \
    --model_path ../.cache \
    --dataset glue --task mnli \
    --deepspeed_config tests/config/zero_stage3.json