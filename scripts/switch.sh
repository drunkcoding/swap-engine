python -m scripts.switch --gin_file=config/t5x/base_eval.gin \
    --gin.MIXTURE_OR_TASK_NAME=\"trivia_qa_open\" \
    --gin.CHECKPOINT_PATH=\"/mnt/xly/checkpoints/t5x/moe/switch_classic/base/e128/checkpoint_550000/\" \
    --gin.EVAL_OUTPUT_DIR=\"outputs\" \
    --gin.NUM_MODEL_PARTITIONS=1