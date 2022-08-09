import torch
import random
import numpy as np
import torch.nn as nn
import cPickle
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

import torch.nn.functional as F
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def Tensor2Varible(tensor_):
    var = Variable(tensor_)
    var = var.cuda() if use_cuda else var
    return var

class KNRM(nn.Module):
    def __init__(self, vocab_size, config):
        super(KNRM, self).__init__()
        self.vocab_size = vocab_size
        self.embsize = config['embsize']

        self.drate = config['drate']

        self.kernel_size = config['kernel_size']
        self.kernel_num = config['kernel_num']
        self.sigma = config['sigma']
        self.exact_sigma = config['exact_sigma']

        self.out_size = config['out_size']

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embsize)

        pre_word_embeds_addr = config['emb'] if 'emb' in config else None

        if pre_word_embeds_addr is not None:
            print('Loading word embeddings')
            pre_word_embeds = cPickle.load(open(pre_word_embeds_addr))
            print('pre_word_embeds size: ',pre_word_embeds.shape)
            self.embedding_layer.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))


        self.score = nn.Linear(self.kernel_num,1)
        self.drop = nn.Dropout(self.drate)
        #self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, input_q, input_d):
        query_mask = 1 - torch.eq(input_q, 0).float()
        sent_mask = 1 - torch.eq(input_d, 0).float()
        input_mask = torch.bmm(query_mask.unsqueeze(-1), sent_mask.unsqueeze(1))
        return input_mask

    def cos_sim(self,q_emb,d_emb):
        q_emb = F.normalize(q_emb, p=2, dim=-1)
        d_emb = F.normalize(d_emb, p=2, dim=-1)
        sim_matrix = torch.bmm(q_emb, d_emb.transpose(1, 2))

        return sim_matrix

    def kernel_layer(self,x,mu,sigma):
        return torch.exp(-0.5 * (x - mu)**2 / (sigma**2))

    def forward(self,query_batch,doc_batch):
        '''
        :param query_variable: (b_s,n_word) padded
        :param document_variable:(b_s,n_word) padded
        :param gt_rels: (b_s,1) float
        :return:
        '''

        query_var = Tensor2Varible(torch.LongTensor(query_batch))
        doc_batch = Tensor2Varible(torch.LongTensor(doc_batch))

        batch_size = doc_batch.size(0)
        doc_batch = doc_batch.view(batch_size,-1)

        q_emb = self.embedding_layer(query_var)
        d_emb = self.embedding_layer(doc_batch)

        cos_sim = self.cos_sim(q_emb,d_emb) #(bs,ql,dl)

        #input_mask = self.get_mask(query_var, doc_batch)
        #cos_sim = cos_sim * input_mask

        KM = []
        mm_doc_sum_list = []
        for i in range(self.kernel_num):
            mu = 1.0 / (self.kernel_num - 1) + (2.0 * i) / (self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            mm_exp = self.kernel_layer(cos_sim,mu,sigma) #* input_mask  #(bs,ql,dl) *
            mm_doc_sum = torch.sum(mm_exp,2) #(bs,ql)

            mm_log = torch.log1p(mm_doc_sum)
            mm_sum = torch.sum(mm_log,1)
            KM.append(mm_sum)

            mm_doc_sum_list.append(mm_doc_sum)
        Phi = torch.stack(KM,1)

        rel_predicted = self.score(Phi)

        return rel_predicted
