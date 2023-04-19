deepspeed --include "localhost:0" tests/python/ds_inference.py \
    --model_name switch-base-8 \
    --model_path /mnt/raid0nvme1/xly/.cache \
    --batch_size 1 --dataset glue --task mnli \
    --deepspeed_config tests/config/zero_stage3.json


deepspeed --num_gpus=1 transformers/examples/pytorch/translation/run_translation.py \
--deepspeed tests/config/ds_config_zero3.json \
--model_name_or_path google/switch-base-128 --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro --cache_dir /mnt/raid0nvme1/xly/.cache/huggingface

deepspeed --num_gpus=1 transformers/examples/pytorch/translation/run_translation.py \
--deepspeed tests/config/zero_stage3.json \
--model_name_or_path google/switch-base-128 --per_device_eval_batch_size 1 \
--output_dir output_dir --overwrite_output_dir \
--do_predict --dataset_name glue --task_name mnli_matched \
--pad_to_max_length True --max_source_length 128 --max_target_length 32 \
--cache_dir /mnt/raid0nvme1/xly/.cache/huggingface



python tests/python/ds_inference.py \
--model_name_or_path google/switch-base-8 --per_device_eval_batch_size 1 \
--output_dir output_dir --overwrite_output_dir \
--do_predict --dataset_name glue --task_name mnli_matched \
--pad_to_max_length True --max_source_length 128 --max_target_length 32 \
--cache_dir /mnt/raid0nvme1/xly/.cache/huggingface

TORCH_EXTENSIONS_DIR=./torch-extensions deepspeed --num_gpus=1 tests/python/ds_inference.py \
--model_name switch-xxl-128 --model_path /mnt/raid0nvme1/xly/.cache \
--batch_size 1 --dataset squad \
--deepspeed_config tests/config/zero_stage3.json

deepspeed --autotuning run --num_gpus=4 tests/python/ds_inference.py \
--model_name switch-base-128 --model_path /mnt/raid0nvme1/xly/.cache/huggingface \
--batch_size 1 --dataset glue --task mnli \
--deepspeed_config tests/config/zero_stage3.json