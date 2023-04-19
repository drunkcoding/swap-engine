

python -m pysrc.moe_glue_triton_client \
    --model_name switch-base-128 \
    --num_processes 1 \
    --batch_size 1 --dataset glue --task mnli

python -m pysrc.moe_glue_triton_client \
    --model_name switch-base-256 \
    --num_processes 1 \
    --batch_size 1 --dataset super_glue --task boolq

python -m pysrc.moe_glue_triton_client \
    --model_name switch-base-256 \
    --num_processes 1 \
    --batch_size 1 --dataset squad

    