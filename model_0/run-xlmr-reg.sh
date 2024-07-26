# sentence-transformers/LaBSE
# xlm-roberta-large
# xlm-roberta-base
# vinai/phobert-base
# lr: 5e-5 default of huggingface
# lr_fc 5e-3, lr_bert 1e-6: 50 epoch 0.7341
# lr_fc 5e-5, lr_bert 5e-5: 50 epoch 0.7161
# train thêm 50 epoch với lr = 5e-5
# Data_Add/train_1000.z
#         --manual_train_file Data_Add/train_1000.z \

# seed 1308 ket qua tot nhat.

# them 400 sample: 0.7261
# them 1000 sample: 

accelerate launch model_0/regression.py \
        --num_labels 6 \
        --model_name_or_path xlm-roberta-base \
        --tokenizer_name xlm-roberta-base \
        --train_file Data/remake_train_1.z \
        --manual_data_file Data_Add/train_1000_1811_large.z \
        --aug_train_folder Data/Aug_Data/ \
        --test_file Data/test_1.z \
        --output_dir model_0/model_regression \
        --max_seq_length 512 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --lr_bert 3e-5 \
        --lr_fc 3e-5 \
        --state 0 \
        --num_train_epochs 150 \
        --freeze_layer_count 0 \
        --best_score 0.73 \
        --add_aug_data 0 \
        --manual_data 0 \
        --num_to_stop_training 20 \
        --seed 1221