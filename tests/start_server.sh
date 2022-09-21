# run deepspped server with default config

CKPT_BASE=/mnt/xly/checkpoints/

LD_LIBRARY_PATH=${HOME}/miniconda3/envs/swap-engine/lib:${LD_LIBRARY_PATH} \
python -m baseline.ds_grpc_server \
    --model_paths=${CKPT_BASE}/vit-tiny-patch16-224,${CKPT_BASE}/vit-small-patch16-224,${CKPT_BASE}/vit-base-patch16-224,${CKPT_BASE}/vit-large-patch16-224 \
    --model_type=vit \
    --deepspeed_config=config/deepspeed/ds_config_stage0.json \