import torch 
from sklearn.metrics import f1_score
import numpy as np 
import joblib
import pandas as pd

def binary_6_to_score(vector):
    res = []
    for i in range(0, len(vector), 6):
        a = vector[i:i+6]
        res.append(a.index(max(a)))
    return res

def convert_logits(logits):
    results = []
    for logit in logits:
        res = []
        for i in range(0, 36, 6):
            x = logit[i:i+6]
            res.append(torch.argmax(x))
        res = torch.stack(res)
        results.append(res)
    results = torch.stack(results)
    return results

def convert_logits_1(logits):
    results = []
    for logit in logits:
        res = []
        for i in range(0, 30, 5):
            x = logit[i:i+5]
            if max(x) < 0.5:
                res.append(torch.tensor(0))
            else:
                res.append(torch.argmax(x)+1)
        res = torch.stack(res)
        results.append(res)
    results = torch.stack(results)
    return results

def convert_logits_loss(logits):
    
    results = []
    for logit in logits:
        res = []
        for i in range(0, 30, 5):
            x = logit[i:i+5]
            idx = torch.max(x) 
            res.append(idx)
        results.append(res)
    results = torch.tensor(results)
    return results    

def calculate_binary_score(labels, preds):
    label_binarys = torch.transpose(torch.tensor([[0 if i ==0 else 1 for i in row] for row in labels]), 0,1)
    preds_binarys = torch.transpose(torch.tensor([[0 if i ==0 else 1 for i in row] for row in preds]),0,1)
    
    labels_1 = torch.transpose(labels, 0, 1)
    preds_1 = torch.transpose(preds, 0, 1)
    score = 0
    for idx in range(len(labels_1)):
        f1_1 = float(f1_score(preds_binarys[idx], label_binarys[idx]))
        score = score + f1_1
    return float(score)/6

def calculate_score(labels, preds):
    label_binarys = torch.transpose(torch.tensor([[0 if i ==0 else 1 for i in row] for row in labels]), 0,1)
    preds_binarys = torch.transpose(torch.tensor([[0 if i ==0 else 1 for i in row] for row in preds]),0,1)
    
    labels_1 = torch.transpose(labels, 0, 1)
    preds_1 = torch.transpose(preds, 0, 1)
    score = 0
    for idx in range(len(labels_1)):
        f1_1 = float(f1_score(preds_binarys[idx], label_binarys[idx]))
#         print(f"f1_1 = {f1_1}")
        #r2 score
        numerator = 0
        denominator = 0
        num_cas = 0
        for j in range(len(label_binarys[idx])):
            if label_binarys[idx][j] ==1 and preds_binarys[idx][j] == 1:
                num_cas +=1
                numerator += (labels_1[idx][j] - preds_1[idx][j])**2
                denominator += 16
        if num_cas == 0:
            r2_2 = 1
        else:
            r2_2 = 1 - numerator/denominator
#         print(f"r2_2 = {r2_2}")
        score = score + f1_1*r2_2
    return float(score)/6

# input: 2 list
def compare_arrays(a, b):
    return (np.array(a)==np.array(b)).all()

def convert_z_to_csv(inputfile, outputfile):
#     datafile = 'Data/test_3.z'
#     outputfile = 'Data/test_3.csv'
    data = joblib.load(inputfile)
    def convert_logit_to_score(logit):
        res = []
        for i in range(0, 36, 6):
            x = logit[i:i+6]
            res.append(x.index(max(x)))
        return res
    res = []
    for i in data:
        x = [i[0]] + convert_logit_to_score(i[1])
        res.append(x)
    df = pd.DataFrame(res, columns=['review', 'giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'van_chuyen', 'mua_sam'])
    df.to_csv(outputfile, index=False)
    
def get_confidence(logit):
    # input: tensor
    conf = []
    with torch.no_grad():
        logit = logit.to('cpu')
        for i in range(0, 36, 6):
            x = logit[i:i+6]    
    #         print(type(x))
            x = round(max(torch.nn.functional.softmax(x, dim=0).tolist()), 4)
            conf.append(x)
    return conf