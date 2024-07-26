python3 DownstreamTask/multi_tpu_core/QA/train_LM_BM25_hardnegs.py \
        --data_path DownstreamTask/data/FiQA/task2 \
        --model_name_or_path bert-base-cased \
        --tokenizer_name bert-base-cased \
        --max_seq_length 256 \
        --per_device_train_batch_size 16 \
        --dataset FiQA_2018 \
        --num_train_epochs 1 \
        --learning_rate 5e-5