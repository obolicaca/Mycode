import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import DistilBertForTokenClassification,DistilBertConfig


class NER(nn.Module):
    """ NER model for tagging """
    def __init__(self,num_labels):
        super(NER, self).__init__()
        self.num_labels = num_labels
        self.config = DistilBertConfig.from_pretrained('distilbert-base-cased',num_labels = self.num_labels) # 导入配置文件
        # 单独指定config，在config中指定分类个数
        self.distilbert = DistilBertForTokenClassification.from_pretrained('distilbert-base-cased',config= self.config)

    def forward(self,input_ids,attention_mask,labels):
        out = self.distilbert(input_ids = input_ids , labels = labels, attention_mask = attention_mask)
        return out

