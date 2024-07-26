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
    RobertaModel,
    MODEL_MAPPING,
    CONFIG_MAPPING,
#     RobertaClassificationHead,
    RobertaPreTrainedModel
)
import logging   
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import random
import copy
from tqdm import tqdm as tqdm
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from prettytable import PrettyTable
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers.utils.versions import require_version
from datasets import load_metric
import accelerate
import utils
import joblib
import os
import sys

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
        "--manual_data_file", type = str, help="A csv or a json file contains manual data."
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
    parser.add_argument(
        "--manual_data",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "--num_to_stop_training",
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


# class ClassificationHead(torch.nn.Module):
#     def __init__(self, config):
#         super(ClassificationHead).__init__()
#         self.num_labels = config.num_labels
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    
#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = F.sigmoid(self.out_proj(x))
#         return x
    

# class JointModel(torch.nn.Module):
#     def __init__(self, config, binary_weight=1., reg_weight=1.):
#         super(JointModel).__init__()
#         self.num_labels = config.num_labels
#         self.embedding_model = AutoModel.from_pretrained(config._name_or_path)
#         self.classifier = ClassificationHead(config)
        
#         self.binary_weight = binary_weight
#         self.reg_weight = reg_weight
        
#     def forward(self, input_ids, attention_mask, binary_labels, reg_labels):
#         outputs = self.embedding_model(input_ids, attention_mask)
#         sequence_output = outputs[0]
#         binary_logits = self.classifier(sequence_output)
#         reg_logits = 5*binary_logits
        

def main():
    
    args = parse_args()
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    
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
    
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()
    
    device = accelerator.device    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)      

    def read_data(data_file):
        if '.csv' in data_file:
            df = pd.read_csv(data_file, delimiter=',', header=0, encoding="utf8")
            data = []
            for sample in df.values.tolist():
                data.append([sample[0], sample[1:]])
            return data
        else:
            return joblib.load(data_file)
        
    def get_aspect_and_score_vector(logit):
        aspect_logit = []
        score_logit = []
        for i in range(0, 36, 6):
            x = logit[i:i+6].index(max(logit[i:i+6]))
            if x > 0:
                aspect_logit.append(1)
            else:
                aspect_logit.append(0)
            score_logit.append(x)
        return aspect_logit, score_logit

    def GenericDataLoader(data, batch_size, max_model_length):
        ids = []
        masks = []
        aspect_labels, score_labels = [], []
        max_length = min(max_model_length, args.max_seq_length)
        for sample in data:
            sent = sample[0]
            inputs = tokenizer(sent, return_tensors="np", padding='max_length', truncation=True, max_length=max_length)
            encoded_sent = inputs['input_ids'][0]
            mask = inputs['attention_mask'][0]
            ids.append(encoded_sent)
            masks.append(mask)
            
            aspect, score = get_aspect_and_score_vector(sample[1])
            aspect_labels.append(aspect)
            score_labels.append(score)
            
        inputs = torch.tensor(np.array(ids))
        masks = torch.tensor(np.array(masks))
        aspect_labels = torch.tensor(np.array(aspect_labels), dtype=torch.float)
        score_labels = torch.tensor(np.array(score_labels), dtype=torch.float)
        data = TensorDataset(inputs, masks, aspect_labels, score_labels)
        
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def evaluation(model, test_dataloader, num_test_sample):

        model.eval()
        targets, preds = [], []
        for batch in tqdm(test_dataloader):

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, _, b_score_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            logits = outputs[0]
            bi_logits = torch.sigmoid(logits)
            reg_logits = 5*bi_logits
            outputs, labels = accelerator.gather([reg_logits, b_score_labels])
            
            targets.append(labels)
            preds.append(outputs)
            
        targets = torch.cat(targets)[:num_test_sample*6]
        preds = torch.cat(preds)[:num_test_sample*6]
#         accelerator.print("target shape: {}".format(targets.shape))
#         accelerator.print("target shape: {}".format(preds.shape))
#         return
        preds = torch.round(preds)
    
        score =  utils.calculate_score(targets, preds)
        
        return score

    def train(model, epochs, bi_loss_fn, reg_loscc_fn, optimizer, lr_scheduler, train_dataloader, val_dataloader=None,  num_val_sample=None):
        best_score = 0
        best_model = model
        for epoch in range(epochs):
            accelerator.print('Training epoch: {}'.format(epoch))    
            total_loss = 0
            model.train()
            
            label_list, pred_list = [], []
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_bi_labels = batch[2].to(device)
                b_reg_labels = batch[3].to(device)
                
                model.zero_grad()
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
                logits = outputs[0]
                
                bi_logits = torch.sigmoid(logits)
                reg_logits = 5*bi_logits
                
                bi_loss = bi_loss_fn(bi_logits, b_bi_labels)
                reg_loss = reg_loss_fn(reg_logits, b_reg_labels)
                
                total_loss = total_loss + bi_loss + reg_loss
                accelerator.backward(bi_loss + reg_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            lr_scheduler.step()
            
            avg_train_loss = total_loss / len(train_dataloader)
            accelerator.print("Average training loss: {0:.4f}".format(avg_train_loss))
  
            accelerator.print("Running validation, epoch: {}".format(epoch))
            
            current_score = evaluation(model, val_dataloader, num_val_sample)
            accelerator.print("Current score: {0:.4f}".format(current_score))
            accelerator.print("Best score: {0:.4f}".format(best_score))
            if current_score > best_score:
                best_model = copy.deepcopy(model)
                best_score = current_score

        return best_model

    # y = torch.tensor([i[1] for i in _train_data[:num_train]])
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    config = model.config
    max_model_length = config.max_position_embeddings

    train_data = read_data(args.train_file)
    if args.add_aug_data:
        for f in os.listdir(args.aug_train_folder):
            train_data.extend(read_data(args.aug_train_folder+f))
    
    if args.manual_data:
        train_data.extend(read_data(args.manual_data_file))
    
    test_data = read_data(args.test_file)

    num_train_sample, num_test_sample = len(train_data), len(test_data)
    train_dataloader = GenericDataLoader(train_data, args.per_device_train_batch_size, max_model_length)
    test_dataloader = GenericDataLoader(test_data, args.per_device_eval_batch_size, max_model_length)

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
#     bert_params = model.roberta.encoder.named_parameters()
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
    num_training_steps = args.num_train_epochs * len(train_dataloader)
    num_warmup_steps = 0
    lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)
#     loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bi_loss_fn = torch.nn.BCEWithLogitsLoss()
    reg_loss_fn = torch.nn.MSELoss()

    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
        )
                              
    # test overfit
    best_model = train(model, args.num_train_epochs, bi_loss_fn, reg_loss_fn, optimizer, lr_scheduler, train_dataloader, train_dataloader, num_train_sample)
    final_score = evaluation(best_model, train_dataloader, num_train_sample)
                              
#     best_model = train(model, args.num_train_epochs, bi_loss_fn, reg_loss_fn, optimizer, lr_scheduler, train_dataloader, test_dataloader, num_test_sample)
#     final_score = evaluation(best_model, test_dataloader, num_test_sample)
    
#     accelerator.print('======================')
#     accelerator.print(" Final score: {0:.4f}".format(final_score))
#     if final_score > args.best_score:
#         accelerator.wait_for_everyone()
#         unwrapped_model = accelerator.unwrap_model(best_model)
#         unwrapped_model.save_pretrained(
#             args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
#         )        
    
if __name__ == "__main__":
    main()
    
    