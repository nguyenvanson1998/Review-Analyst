import torch

def convert_logit(logit):
    res = []
    for i in range(0, 36, 6):
        x = logit[i:i+6]
        res.append(torch.argmax(x))
    res = torch.stack(res)
    return res