import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import *

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.bert_model = BertModel.from_pretrained(config['BERT_folder'])
        self.lin_out = nn.Linear(config['text_dim'], 1)
        self.output_layer_index = config['output_layer_index']

    def forward(self, text_batch):
        text_batch = Tensor2Varible(torch.LongTensor(text_batch))
        #segments_ids = Tensor2Varible(torch.LongTensor(seq_ids))
        segments_ids = Tensor2Varible(torch.zeros(text_batch.size(), dtype=torch.long))

        encoded_layers, _ = self.bert_model(text_batch, segments_ids)
        embed = encoded_layers[self.output_layer_index][:, 0, :]#.detach()
        dm_score = self.lin_out(embed)

        return dm_score
