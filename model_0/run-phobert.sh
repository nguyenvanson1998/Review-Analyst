# sentence-transformers/LaBSE
# xlm-roberta-large
# xlm-roberta-base
# vinai/phobert-base
# lr: 5e-5 default of huggingface

accelerate launch model_0/classification.py \
        --num_labels 36 \
        --model_name_or_path vinai/phobert-base \
        --tokenizer_name vinai/phobert-base \
        --output_dir model_0/model-phobert \
        --train_file Data_seg/clean-train.z \
        --aug_train_folder Data_seg/Aug_Data/ \
        --test_file Data_seg/clean-public.z \
        --max_seq_length 512 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --lr_bert 1e-6 \
        --lr_fc 5e-3 \
        --state 0 \
        --num_train_epochs 50 \
        --freeze_layer_count 0 \
        --best_score 0.7341 \
        --add_aug_data 1 \
        --num_to_stop_training 10