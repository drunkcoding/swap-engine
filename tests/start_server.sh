# run deepspped server with default config

CKPT_BASE=/mnt/xly/checkpoints

LD_LIBRARY_PATH=${HOME}/miniconda3/envs/swap-engine/lib:${LD_LIBRARY_PATH} \
python -m baseline.ds_grpc_server \
    --model_paths=${CKPT_BASE}/vit-xl-patch16-224 \
    --model_type=vit \
    --deepspeed_config=config/deepspeed/ds_config_stage3.json \
    --cfg_only