import torch
import torch.nn as nn
import torch.nn.functional as F



class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)
    
class CrossEntropyLoss2dv2(nn.Module):
    
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2dv2, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
