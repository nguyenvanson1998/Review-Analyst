from transformers import AutoModelForSequenceClassification, AutoTokenizer
import utils
import processing
import config
import torch

import torch.nn as nn
from transformers import XLMRobertaModel, RobertaPreTrainedModel, XLMRobertaForSequenceClassification
import torch
class RobertaClassificationLSTM(nn.Module):
    """ Head for sentence-level classification tasks."""
    def __init__(self,config):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_1 = nn.Linear(config.hidden_size, 256)
        self.out_proj = nn.Linear(256, config.num_labels)

        self.dropout = nn.Dropout(classifier_dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.dense_1(x)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.out_proj(x)
        return x
      
class SequenceClassification(XLMRobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config):
        super().__init__(config)
        self.classifier = RobertaClassificationLSTM(config)

class ClassifyReviewSolver:
    def __init__(self, config):
        # self.classify_model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=config.NUM_LABEL)
        self.classify_model = SequenceClassification.from_pretrained(config.MODEL_PATH, num_labels=config.NUM_LABEL)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    def solve(self, text):
        self.classify_model.eval()
        text = processing.preprocessing_sample(text)
        input = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True)
        with torch.no_grad():
            logit = self.classify_model(**input)[0][0]
        predict_results = utils.convert_logit(logit).tolist()
        return predict_results

# class ClassifyReviewSolver:
#     def __init__(self, config):
#         self.classify_model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_PATH)
#         self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

#     def solve(self, text):
#         text = processing.preprocessing_sample(text)
#         input = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=config.MAX_LEN)
#         logit = self.classify_model(**input)[0][0]
#         predict_results = torch.round(5*torch.sigmoid(logit)).tolist()
#         return [int(i) for i in predict_results]