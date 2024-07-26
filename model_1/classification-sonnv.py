import sys
sys.path.append('/home/caohainam/Review-Analytics')
import argparse
from unittest.util import _MAX_LENGTH
import pandas as pd
import transformers
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoModel, 
    BertModel,
    MODEL_MAPPING,
    CONFIG_MAPPING
)
import logging   
from unidecode import unidecode
# from transformers import TrainingArguments
import numpy as np
# from datasets import load_metric, load_dataset
# from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import random
import copy
from tqdm import tqdm as tqdm
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from prettytable import PrettyTable
from accelerate import Accelerator
from transformers.utils.versions import require_version
from datasets import load_metric
import accelerate
import utils
import joblib
import os
import sys
import datetime
import torch.nn.functional as F

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--aug_train_folder", type = str, help="A csv or a json file containing the all data, split to train and test model."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--num_labels", type=int, required=True, help="number of labels"
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size"
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='xlm-roberta-base',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial common learning rate for model",
    )
    parser.add_argument(
        "--lr_bert",
        type=float,
        default=5e-5,
        help="Initial learning rate for bert layer",
    )
    parser.add_argument(
        "--lr_fc",
        type=float,
        default=5e-5,
        help="Initial learning rate for fully connected layer",
    )
    parser.add_argument(
        "--state",
        type=int,
        default=None,
        help="train full or test model flow",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--freeze_layer_count", type=int, default=None, help="Freeze layer in bert model")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help=""
    )
    parser.add_argument(
        "--best_score",
        type=float,
        help=""
    )
    parser.add_argument(
        "--add_aug_data",
        type=int,
        default=None,
        help="",
    )
    
    args = parser.parse_args()

    return args
# FinancialPhraseBank, CausalityDetection, Lithuanian
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
    
    args = parse_args()
#     accelerator = Accelerator()
#     try:
#         device = accelerator.device
#     except:
#         device = 'cpu'
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device(device)
#     print('device: {}'.format(device))

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    
#     if args.aug_train_folder:
#         accelerator.print(1)
#     else:
#         accelerator.print(0)
#     sys.exit()
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
#         datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
#     print(1)
#     sys.exit()
#     try:
#         device = accelerator.device
#     except:
#         device = 'cpu'

    accelerator.wait_for_everyone()
    
    device = accelerator.device
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(device)
    # print('device: {}'.format(device))
    # sys.exit()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)      
    num_epochs = args.num_train_epochs
    # result_file = 'Evaluation'+args.dataset+'task.txt'
    # f = open(result_file, 'w')

    def read_data(data_file):
        if '.csv' in data_file:
            df = pd.read_csv(data_file, delimiter=',', header=0, encoding="utf8")
            data = []
            for sample in df.values.tolist():
                data.append([sample[0], sample[1:]])
            return data
        else:
            return joblib.load(data_file)

    def GenericDataLoader(data, batch_size):
        
        def convert_label(logit):
            res = []
            for i in range(0, 36, 6):
                x = logit[i:i+6][1:]
                res.extend(x)
            return res
        
        ids = []
        masks = []
        for sample in data:
            sent = sample[0]
            inputs = tokenizer(sent, return_tensors="np", padding='max_length', truncation=True, max_length=args.max_seq_length)
            encoded_sent = inputs['input_ids'][0]
            mask = inputs['attention_mask'][0]
            ids.append(encoded_sent)
            masks.append(mask)
        inputs = torch.tensor(np.array(ids))
        masks = torch.tensor(np.array(masks))
        labels = torch.tensor(np.array([convert_label(i[1]) for i in data]), dtype=torch.float)
        data = TensorDataset(inputs, masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluation(model, test_dataloader, num_test_sample):
        
        model.eval()
        targets, preds = [], []
        for batch in tqdm(test_dataloader):

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            logits = torch.sigmoid(outputs[0])
            outputs, labels = accelerator.gather([logits, b_labels])
            
            targets.append(labels)
            preds.append(outputs)
            
        targets = torch.cat(targets)[:num_test_sample*30]
        preds = torch.cat(preds)[:num_test_sample*30]
        
        targets = utils.convert_logits_1(targets)
        preds = utils.convert_logits_1(preds)
#         accelerator.print("target shape: {}".format(targets.shape))
#         accelerator.print("target shape: {}".format(preds.shape))
#         return 
        score =  utils.calculate_score(targets, preds)
        return score

    def train(model, epochs, loss_fn_1, loss_fn_2, optimizer, lr_scheduler, train_dataloader, val_dataloader=None,  num_val_sample=None):
        best_score = 0
        best_model = model
        for epoch in range(epochs):
            accelerator.print('Training epoch: {}'.format(epoch))    
            # f.writelines('Training epoch: {}\n'.format(epoch))    
            total_loss = 0
            model.train()
            
            label_list, pred_list = [], []
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
                logits = outputs[0]
                # accelerator.print(logits)
                # accelerator.print(b_labels)
                loss_1 = loss_fn_1(logits, b_labels)
                loss_2 = loss_fn_2(utils.convert_logits_loss(logits).to(device), 
                                   utils.convert_logits_loss(b_labels).to(device))
#                 total_loss += loss.item()
                total_loss += loss_1 + loss_2
                accelerator.backward(loss_1+loss_2)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
#                 break
                
            lr_scheduler.step()
#         return model
            
            avg_train_loss = total_loss / len(train_dataloader)
            accelerator.print("Average training loss: {0:.4f}".format(avg_train_loss))
            # f.writelines("Average training loss: {0:.4f}".format(avg_train_loss)+'\n')
  
            accelerator.print("Running validation, epoch: {}".format(epoch))
#             f.writelines("Running validation, epoch: {}\n'".format(epoch))
#             accelerator.print(classification_report(targets, preds, zero_division=0, digits=4))
#             f.writelines(classification_report(targets, preds, zero_division=0, digits=4)+'\n')
#             sys.exit()
            
            current_score = evaluation(model, val_dataloader, num_val_sample)
            accelerator.print("Current score: {0:.4f}".format(current_score))
            accelerator.print("Best score: {0:.4f}".format(best_score))
            if current_score > best_score:
                best_model = copy.deepcopy(model)
                best_score = current_score

        return best_model

    # y = torch.tensor([i[1] for i in _train_data[:num_train]])

    train_data = read_data(args.train_file)
    if args.aug_train_folder:
        for f in os.listdir(args.aug_train_folder):
            train_data.extend(read_data(args.aug_train_folder+f))
    test_data = read_data(args.test_file)
    
    if args.state:
        train_data = train_data[:400]
        test_data = test_data[:200]
    # num_train = int(len(train_data)*0.8)

    num_train_sample, num_test_sample = len(train_data), len(test_data)
    train_dataloader = GenericDataLoader(train_data, args.per_device_train_batch_size)
    test_dataloader = GenericDataLoader(test_data, args.per_device_eval_batch_size)
#     labels = [i[1] for i in train_data]
#     label_count = [sum(i) for i in zip(*labels)]
#     pos_weight = torch.tensor([i/num_train_sample for i in label_count])

#     y = torch.tensor([i[1] for i in train_data])
#     class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
#     class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)

#     embedding_model = BertModel.from_pretrained(args.model_name_or_path).to(device)
    # sys.exit()
#     if args.freeze_layer_count:
#         # We freeze here the embeddings of the model
#         for param in embedding_model.embeddings.parameters():
#             param.requires_grad = False

#         if args.freeze_layer_count != -1:
#             # if freeze_layer_count == -1, we only freeze the embedding layer
#             # otherwise we freeze the first `freeze_layer_count` encoder layers
#             for layer in embedding_model.encoder.layer[:args.freeze_layer_count]:
#                 for param in layer.parameters():
#                     param.requires_grad = False

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    
    if args.freeze_layer_count:
        # We freeze here the embeddings of the model
        if 'roberta' in str(type(model)):
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
        else:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        if args.freeze_layer_count != -1:
            # if freeze_layer_count == -1, we only freeze the embedding layer
            # otherwise we freeze the first `freeze_layer_count` encoder layers
            if 'roberta' in str(type(model)):
                for layer in model.roberta.encoder.layer[:args.freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for layer in model.bert.encoder.layer[:args.freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False
            
    """
    optimizer
    """
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if 'roberta' in str(type(model)):
        bert_params = model.roberta.encoder.named_parameters()
    else:
        bert_params = model.bert.encoder.named_parameters()
    classifier_params = model.classifier.named_parameters()
    grouped_params = [
        {'params': [p for n,p in bert_params if p.requires_grad==True], 'lr': args.lr_bert},
        {'params': [p for n,p in classifier_params if p.requires_grad==True], 'lr': args.lr_fc}
    ]

    optimizer = torch.optim.AdamW(grouped_params)

    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = 20 * len(train_dataloader)
    lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)
#     loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn_1 = torch.nn.BCEWithLogitsLoss()
    loss_fn_2 = torch.nn.BCEWithLogitsLoss()

    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
        )
                              
    # test overfit
#     best_model = train(model, args.num_train_epochs, loss_fn, optimizer, lr_scheduler, train_dataloader, train_dataloader, num_train_sample)
#     final_score = evaluation(best_model, train_dataloader, num_train_sample)
                              
    best_model = train(model, args.num_train_epochs, loss_fn_1, loss_fn_2, optimizer, lr_scheduler, train_dataloader, test_dataloader, num_test_sample)
    final_score = evaluation(best_model, test_dataloader, num_test_sample)
    
    accelerator.print('======================')
    accelerator.print(" Final score: {0:.4f}".format(final_score))
    if final_score > args.best_score:
        accelerator.wait_for_everyone()
    #     output_dir = '_'.join([args.output_dir, str(datetime.now())])
        unwrapped_model = accelerator.unwrap_model(best_model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
    
    
#     acc, mac_f1_score, mic_f1_score = evaluation(best_model, test_dataloader, num_test_sample)

#     accelerator.print('======================')
#     accelerator.print(" Final acccuracy: {0:.4f}".format(acc))
#     accelerator.print(" Final macro f1 score: {0:.4f}".format(mac_f1_score))
#     accelerator.print(" Final micro f1 score: {0:.4f}".format(mic_f1_score))

    # f.close()
    
if __name__ == "__main__":
    main()
    
    