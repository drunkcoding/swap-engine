python -m scripts.switch --gin_file=config/t5x/large/e128/base_eval.gin \
    --gin.MIXTURE_OR_TASK_NAME=\"trivia_qa_open\" \
    --gin.CHECKPOINT_PATH=\"/mnt/raid0nvme1/xly/checkpoints/t5x/moe/switch_classic/large/e128/checkpoint_483100\" \
    --gin.EVAL_OUTPUT_DIR=\"outputs\" \
    --gin.NUM_MODEL_PARTITIONS=1