# bert-base-cased
# bert-base-multilingual-cased
accelerate launch DownstreamTask/multi_tpu_core/classification.py \
        --num_labels 2 \
        --model_name_or_path english_finance_15GB/flax_model.msgpack \
        --dataset CausalityDetection \
        --train_file DownstreamTask/data/Causality\ Detection/fnp2020-task1-train.csv \
        --test_file DownstreamTask/data/Causality\ Detection/fnp2020-task1-test.csv \
        --tokenizer_name bert-base-multilingual-cased \
        --max_seq_length 512 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --lr_bert 5e-6 \
        --lr_fc 1e-3 \
        --state 1 \
        --num_train_epochs 5 \
        --freeze_layer_count 0 \
        --retrain_model 1\
        --config english_finance_15GB/config.json