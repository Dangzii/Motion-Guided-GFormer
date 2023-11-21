import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self,in_feat=512,num_classes=300,dropout_rate=0.5,weight=1.0,fc=False,label_smooth=None):
        super(CrossEntropy,self).__init__()
        if fc:
            self.linear = nn.Linear(in_feat,num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight
        self.fc = fc
        self.label_smooth = label_smooth
        self.class_num = num_classes
    
    def forward(self,loss,predicts,targets=None):
        if self.fc:
            predicts = self.linear(self.dropout(predicts))

        if self.training:
            if self.label_smooth is not None:
                logprobs = F.log_softmax(predicts, dim=1)	# softmax + log
                target = F.one_hot(targets, self.class_num)	# for one-hot
                target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
                loss = -1*torch.sum(target*logprobs, 1)
                return loss.mean()
            else:
                return self.criterion(predicts,targets)
        else:
            return predicts, self.criterion(predicts,targets)
