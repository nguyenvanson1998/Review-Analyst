# bert-base-cased
# bert-base-multilingual-cased
accelerate launch DownstreamTask/multi_tpu_core/regression.py \
        --k_fold 10 \
        --model_name_or_path english_finance_15GB_3/flax_model.msgpack \
        --validation_strategy cross_validation \
        --data_file DownstreamTask/data/FiQA/task1/train/task1_headline_ABSA_train.json DownstreamTask/data/FiQA/task1/train/task1_post_ABSA_train.json \
        --tokenizer_name bert-base-multilingual-cased \
        --max_seq_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 16 \
        --optim_strategy 0 \
        --learning_rate 5e-4 \
        --lr_bert 5e-6 \
        --lr_fc 5e-4 \
        --num_train_epochs 30 \
        --freeze_layer_count 0\
        --patience 1000 \
        --retrain_model 1