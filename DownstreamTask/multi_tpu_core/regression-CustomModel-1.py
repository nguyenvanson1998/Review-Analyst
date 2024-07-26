import argparse
from unittest.util import _MAX_LENGTH
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoModel, 
    BertModel,
    MODEL_MAPPING,
    CONFIG_MAPPING
)
import transformers
from unidecode import unidecode
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric, load_dataset
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import random
from tqdm import tqdm_notebook, tqdm
import copy
from tqdm.auto import tqdm
import json
import sys
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import math
from accelerate import Accelerator
from sklearn.metrics import r2_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print('device: {}'.format(device))

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--data_file", type = str, nargs = '+', help="A csv or a json file containing the all data, split to train and test model."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--validation_strategy", type=str, default=None, help="cross validation or train-test validation"
    )
    parser.add_argument(
        "--k_fold", type=int, default=None, help="parameter for cross validation"  
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="batch size"
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
        help="Initial learning rate (after the potential warmup period) to use.",
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
    # parser.add_argument(
    #     "--lr_scheduler_type",
    #     type=SchedulerType,
    #     default="linear",
    #     help="The scheduler type to use.",
    #     choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    # )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default=None,
    #     help="Model type to use if training from scratch.",
    #     choices=MODEL_TYPES,
    # )
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
    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         if extension not in ["csv", "json", "txt"]:
    #             raise ValueError("`train_file` should be a csv, json or txt file.")
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         if extension not in ["csv", "json", "txt"]:
    #             raise ValueError("`validation_file` should be a csv, json or txt file.")

    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args
# FinancialPhraseBank, CausalityDetection, Lithuanian

class RegressionBasedBert(torch.nn.Module):

    def __init__(self, embedding_model):
        super(RegressionBasedBert, self).__init__()
        self.embedding_model = embedding_model
        self.linear1 = torch.nn.Linear(768, 384)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(384, 1)

    def forward(self, ids, mask):
        x = self.embedding_model(ids, mask)['last_hidden_state'][:, 0, :]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def main():
    args = parse_args()
    # print(args.data_file)
    org_all_data = []
    for data_file in args.data_file:
        with open(data_file, 'r', encoding='utf-8') as f:
            x = json.load(f)
            for i in list(x.keys()):
                org_all_data.append(x[i])
    
    """
    gen data: [sentence, sentiment score]
    """
    f = open('EvaluationRegressionTask.txt', 'w')
    all_data = []
    for i in org_all_data:
        # print(i)
        sent = i['sentence']
        sentiment_score = float(i['info'][0]['sentiment_score'])
        # print(type(sentiment_score))
        # sys.exit()
        all_data.append([sent, sentiment_score])
    
    # print(len(all_data))
    # print(all_data[:10])
    # train_data = all_data[:1000]
    # test_data = all_data[1000:1100]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    loss_fn = torch.nn.MSELoss()
    """
    data: List -> [sent: string, sentiment_score: float]
    """
    def GenericDataLoader(data):
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
        labels = torch.tensor(np.array([i[1] for i in data], dtype='f'))
        # print(type(labels))
        # sys.exit()
        data = TensorDataset(inputs, masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)
        return dataloader

    def train(model, epochs, train_dataloader, patience = 3):
        the_last_loss = 1e10
        trigger_times = 0
        for epoch in range(epochs):
            print('\n')
            model.train()
            total_loss = 0
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                outputs = model(b_input_ids, b_input_mask)
                outputs = outputs.squeeze(1)
                loss = loss_fn(outputs, b_labels)
                total_loss += loss.item()
        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Early stopping
            the_current_loss = total_loss
            print('Epoch: {}, Total loss: {:.2f}'.format(epoch, total_loss))
            f.writelines('Epoch: {}, Total loss: {:.2f}\n'.format(epoch, the_current_loss))

            # print('The current loss:', the_current_loss)

            if the_current_loss > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)
                f.writelines('trigger times: {}\n'.format(trigger_times))

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    f.writelines('Early stopping!\nStart to test process.\n')
                    return model

            else:
                print('trigger times: 0')
                f.writelines('trigger times: 0\n')
                trigger_times = 0

            the_last_loss = the_current_loss

        return model

    def evaluation(model, test_dataloader):
        targets, preds = [], []
        print('Evaluation')
        f.writelines('Evaluation\n')
        model.eval()
        for step, batch in tqdm(enumerate(test_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                outputs = model(b_input_ids, b_input_mask)
            outputs = outputs.squeeze(1)
            targets.append(b_labels) 
            preds.append(outputs)

        targets = torch.cat(targets)
        preds = torch.cat(preds)
        # print('targets size: {}'.format(targets.shape))
        # print('preds size: {}'.format(preds.shape))
        mse = abs(mean_squared_error(targets.cpu().numpy(), preds.cpu().numpy()))
        r2 = r2_score(targets.cpu().numpy(), preds.cpu().numpy())
        
        print('mse: {:.4f}'.format(mse))
        print('r2: {:4f}'.format(r2))
        f.writelines('mse: {:.4f}\n'.format(mse))
        f.writelines('r2: {:4f}\n'.format(r2))

        return mse, r2

    all_data = pd.DataFrame(all_data)
    kf = KFold(n_splits=args.k_fold, random_state=231, shuffle=True)
    MSE, R2 = [], []
    count = 1
    for train_index, val_index in kf.split(all_data):
        print('=======Start fold: {}=========\n'.format(count))
        f.writelines('=======Start fold: {}=========\n\n'.format(count))
        train_df = all_data.iloc[train_index]
        test_df = all_data.iloc[val_index]
        # print(train_df.head())
        # print(test_df.head())
        # print(len(train_df))
        # print(len(test_df))

        train_data = train_df.values.tolist()
        test_data = test_df.values.tolist()
        # print(train_data[:10])
        # print(test_data[:10])
        
        # sys.exit()


        train_dataloader, test_dataloader = GenericDataLoader(train_data), GenericDataLoader(test_data)
        embedding_model = AutoModel.from_pretrained(args.model_name_or_path).to(device)
        
        if args.freeze_layer_count:
                # We freeze here the embeddings of the model
                for param in embedding_model.embeddings.parameters():
                    param.requires_grad = False

                if args.freeze_layer_count != -1:
                    # if freeze_layer_count == -1, we only freeze the embedding layer
                    # otherwise we freeze the first `freeze_layer_count` encoder layers
                    for layer in embedding_model.encoder.layer[:args.freeze_layer_count]:
                        for param in layer.parameters():
                            param.requires_grad = False
        # for ids, mask, label in train_dataloader:
        #     print(embedding_model(ids.to(device), mask.to(device)))
        #     break
        model = RegressionBasedBert(embedding_model).to(device)
        # best_model = model
        # for ids, mask, label in train_dataloader:
        #     print(model(ids.to(device), mask.to(device)))
        #     sys.exit()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 

        # for epoch in range(args.num_train_epochs):
        #     train(model, train_dataloader)

        model = train(model, args.num_train_epochs, train_dataloader)

        # evaluation(model, test_dataloader)
        # sys.exit()
        mse, r2 = evaluation(model, test_dataloader) 
        MSE.append(mse)
        R2.append(r2)
        # sys.exit()
        print('Finish fold: {}\n\n'.format(count))
        f.writelines('Finish fold: {}\n\n\n'.format(count))
        count += 1
    print('\n\n\n\n')
    print('Final evaluation')
    print('Average mse: {:.2f}'.format(sum(MSE)/args.k_fold))
    print('Average r2: {:.2f}'.format(sum(R2)/args.k_fold))
    f.writelines('\n\n\n\n\n')
    f.writelines('Final evaluation\n')
    f.writelines('Average mse: {:.2f}\n'.format(sum(MSE)/args.k_fold))
    f.writelines('Average r2: {:.2f}\n'.format(sum(R2)/args.k_fold))
    f.close()

if __name__ == "__main__":
    main()