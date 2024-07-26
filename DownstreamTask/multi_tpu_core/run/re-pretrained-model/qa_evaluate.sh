accelerate launch DownstreamTask/multi_tpu_core/evaluate_FiQA.py \
        --data_path DownstreamTask/data/FiQA/task2 \
        --dataset FiQA_2018 \
        --model_name_or_path output/bert-base-cased-v1-FiQA_2018 \
        --tokenizer_name bert-base-cased \
        --max_seq_length 256 \
        --per_device_eval_batch_size 32