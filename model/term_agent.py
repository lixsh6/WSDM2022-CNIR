# -*- coding: utf-8 -*
'''PyTorch class for RL agent to expand terms'''
import torch
from torch import nn
from torch.nn import functional as F
from utils import *
from metrics.rank_evaluations import *
from ranker.bm25 import *
from ranker.knrm import *
from ranker.bert import *
import cPickle

class TermAgent(nn.Module):
    def __init__(self,vocab_size, config):
        super(TermAgent, self).__init__()
        self.vocab_size = vocab_size
        self.we_size = config['word_emb_size']
        self.qe_size = config['q_embed_size']
        self.sample_size = config['sample_size']
        self.term_embedding = nn.Embedding(self.vocab_size, self.we_size)
        self.load_embedding(self.term_embedding, config['word_emb'])

        self.random_mode = config['random_mode']
        self.query_encoder = CNN1D(self.we_size, self.qe_size, config['kernel_size'],config['pool_size_q'])
        #self.cand_term_length = config['cand_term_length']
        #self.decoding_mode = config['decoding_mode']

        self.ind_output_layer = nn.Sequential(nn.Linear(self.we_size + self.qe_size, self.qe_size),
                                        nn.Tanh(),
                                        nn.Linear(self.qe_size,1,bias=False),
                                        nn.Softmax(dim=1))


        self.baseline_layer = nn.Sequential(nn.Linear(self.we_size + self.qe_size, self.qe_size),
                                              nn.Tanh(),
                                              nn.Linear(self.qe_size, 1, bias=False),
                                              nn.Softmax(dim=1))



    def load_embedding(self,emb,addr,name='word embedding'):
        print('=========loading ' + name + ' =========')
        pre_embeds = cPickle.load(open(addr))
        print('Embedding size: ',emb.weight.size())
        print('Original size: ', pre_embeds.shape)
        emb.weight = nn.Parameter(torch.FloatTensor(pre_embeds))
        print('========= Embedding loaded =========')

    def forward(self,query_batch,cand_term_batch,num_repeat=1):
        '''
        :input: (query,candidate_terms)
        :return: (select_probs of candidate_terms)
        '''
        #query_var = Tensor2Varible(torch.LongTensor(query_batch))
        if self.training:
            query_batch = query_batch.repeat(num_repeat,1)
            cand_term_batch = cand_term_batch.repeat(num_repeat,1)

        q_emb = self.term_embedding(query_batch)
        cand_term_mask = 1.0 - torch.eq(cand_term_batch, 0).float()
        cand_term_emb = self.term_embedding(cand_term_batch)

        query_embed = self.query_encoder(q_emb)
        cand_probs, cand_indices = self.select_action(query_embed, cand_term_emb,cand_term_mask)

        #avg_reward = self.baseline_model(query_embed, cand_term_emb)
        expand_idx = torch.gather(cand_term_batch, 1, cand_indices)

        return cand_probs, expand_idx, 0


    def select_action(self,query_embed,cand_term_emb,cand_term_mask, mode='ind'):
        #:param mode:query_embed：(bs,qes),cand_term_emb:(bs,cs,wes)
        #ind: independent selection, output the probs of all cand_terms
        #beam: beam search, sequentially outputs until a special token
        #:return:

        cand_term_length = cand_term_emb.size(1)
        query_embed_r = query_embed.unsqueeze(1).repeat(1, cand_term_length, 1)
        cancate_embed = torch.cat([query_embed_r,cand_term_emb],dim=-1) #(bs,cs,(qes + wes))
        cand_probs = self.ind_output_layer(cancate_embed).squeeze()#(bs,cs,1)
        #cand_probs = cand_probs * cand_term_mask #TODO important!!!
        select_probs, actions = self.sample_action(cand_probs,topN=self.sample_size)
        return select_probs, actions
    
    def sample_action(self,probs,topN=5):
        '''
        :param probs: (bs,cs)
        :param topN: ensure that topN is smaller than the valid cand term length
        :return:
        '''
        if self.random_mode:
            probs = Tensor2Varible(torch.randn(probs.size()))
            select_probs, action = torch.topk(probs, topN, dim=1)
            return select_probs, action
        '''
        rand_probs = Tensor2Varible(torch.randn(probs.size()))
        _, action = torch.topk(rand_probs, topN, dim=1)
        select_probs = torch.gather(probs, 1, action)
        return select_probs, action
        '''
        if self.training:
            probs = probs + 1e-9
            action = probs.multinomial(topN)
            action = Tensor2Varible(action.data)
            select_probs = torch.gather(probs, 1, action)
        else:
            select_probs, action = torch.topk(probs, topN, dim=1)
            #action = action.view(-1, 1)
        #print('select_probs: ', select_probs)
        #print('action: ', action)

        return select_probs, action

    def baseline_model(self,query_embed,cand_term_emb):
        avg_cand_term_emb = torch.mean(cand_term_emb,dim=1)
        cancate_embed = torch.cat([query_embed, avg_cand_term_emb], dim=-1)#(bs,(qes + wes))
        avg_reward = self.baseline_layer(cancate_embed)
        return avg_reward

