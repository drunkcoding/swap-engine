TORCH_EXTENSIONS_DIR=./torch-extensions deepspeed --include="localhost:2,3" tests/python/ds_inference.py \
--model_name switch-large-128 --model_path /mnt/data/xly/.cache \
--batch_size 8 --dataset glue --task mnli \
--deepspeed_config tests/config/ds_config_zero3_cpu.json