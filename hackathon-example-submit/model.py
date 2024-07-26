# import torch.nn as nn
# import torch

# class BERT_REGRESSION(nn.Module):
#     def __init__(self, bert_model, num_labels):
#         super(BERT_REGRESSION, self).__init__()
#         self.num_labels = num_labels
#         self.bert = bert_model
#         self.dropout = nn.Dropout(0.2)
#         self.classifier = nn.Linear(768, num_labels)

#     def forward_custom(self, input_ids, attention_mask=None, labels=None):
#         outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask)
#         sequence_output = self.dropout(outputs[0])
#         logits = self.classifier(sequence_output)[:,0,:]

#         if labels is not None:
#             loss_fct = nn.MSELoss()
#             loss = loss_fct(logits, labels)
#             return logits,loss
#         return logits


# class ReviewClassifierModel(nn.Module):

#     def __init__(self, base_model, num_labels, model_path):
#         super(ReviewClassifierModel, self).__init__()
#         self.num_labels = num_labels
#         self.model = None
#         self.setup(base_model, model_path)

#     def setup(self, base_model, model_path):
#         self.model = BERT_REGRESSION(bert_model=base_model,
#                                        num_labels=self.num_labels)
#         self.model.load_state_dict(
#             torch.load(model_path,
#                        map_location=torch.device('cpu')),
#             strict=False)
#         self.model.to("cpu")

#     def predict(self, ids_tensor, mask_tensor):
#         self.model.eval()
#         with torch.no_grad():
#             results = self.model.forward_custom(ids_tensor, mask_tensor)
#             output = torch.round(torch.clamp(results[0], 0, 5))
#         return output.cpu().numpy().tolist()
