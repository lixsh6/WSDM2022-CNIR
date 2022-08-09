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

class KnowledgeAgent(nn.Module):
    def __init__(self,vocab_size,ent_size,use_knowledge, config):
        super(KnowledgeAgent, self).__init__()
        self.vocab_size = vocab_size
        self.ent_size = ent_size
        self.we_size = config['word_emb_size']
        self.qe_size = config['q_embed_size']
        self.sample_size = config['sample_size']

        self.use_knowledge = use_knowledge

        emb_size = vocab_size + ent_size if self.use_knowledge else vocab_size
        self.embedding_layer = nn.Embedding(emb_size, self.we_size)
        self.load_embedding(self.embedding_layer, config['word_emb'], config['ent_emb_addr'])

        self.random_mode = config['random_mode']
        self.query_encoder = CNN1D(self.we_size, self.qe_size, config['kernel_size'],config['pool_size_q'])

        self.ind_output_layer = nn.Sequential(nn.Linear(self.we_size + self.qe_size, self.qe_size),
                                        nn.Tanh(),
                                        nn.Linear(self.qe_size,1,bias=False),
                                        )
        nn.init.xavier_uniform_(self.ind_output_layer[0].weight)
        nn.init.xavier_uniform_(self.ind_output_layer[2].weight)
        self.ind_output_layer[0].bias.data.fill_(0)
        self.softmax = nn.Softmax(dim=1)


    def load_embedding(self,emb,addr,ent_addr):
        print('=========loading word embedding =========')
        pre_embeds = cPickle.load(open(addr))
        if self.use_knowledge:
            print('=========loading entity embedding =========')
            ent_embeds = cPickle.load(open(ent_addr))
            pre_embeds = np.concatenate((pre_embeds, ent_embeds), axis=0)
        emb.weight = nn.Parameter(torch.FloatTensor(pre_embeds))
        print('Embedding size: ', emb.weight.size())
        print('========= Embedding loaded =========')

    def forward(self,query_ent_idx_batch,cand_term_batch,num_repeat=1):

        if self.training:
            query_ent_idx_batch = query_ent_idx_batch.repeat(num_repeat,1)
            cand_term_batch = cand_term_batch.repeat(num_repeat,1)

        q_emb = self.embedding_layer(query_ent_idx_batch)
        cand_term_mask = 1.0 - torch.eq(cand_term_batch, 0).float()
        cand_term_emb = self.embedding_layer(cand_term_batch)

        query_embed = self.query_encoder(q_emb)
        cand_probs, select_probs, actions, entropy = self.select_action(query_embed, cand_term_emb,cand_term_mask)

        expand_idx = torch.gather(cand_term_batch, 1, actions)

        return cand_probs, select_probs, expand_idx, entropy


    def select_action(self,query_embed,cand_term_emb,cand_term_mask, mode='ind'):
        cand_term_length = cand_term_emb.size(1)
        query_embed_r = query_embed.unsqueeze(1).repeat(1, cand_term_length, 1)
        cancate_embed = torch.cat([query_embed_r,cand_term_emb],dim=-1) #(bs,cs,(qes + wes))

        cand_probs = self.ind_output_layer(cancate_embed).squeeze()#(bs,cs,1)
        cand_probs = self.softmax(cand_probs)
        cand_probs = cand_probs * cand_term_mask # important!!!
        select_probs, actions, entropy = self.sample_action(cand_probs,topN=self.sample_size)
        return cand_probs, select_probs, actions, entropy

    def sample_action(self,probs,topN=5):
        '''
        :param probs: (bs,cs)
        :param topN: ensure that topN is smaller than the valid cand term length
        :return:
        '''
        if self.random_mode:
            probs = Tensor2Varible(torch.randn(probs.size()))
            select_probs, action = torch.topk(probs, topN, dim=1)
            entropy = - torch.sum(probs * probs.log())
            return select_probs, action, entropy

        if self.training:

            probs = probs + 1e-12
            action = probs.multinomial(topN)
            action = Tensor2Varible(action.data)#action = torch.ones_like(action)
            select_probs = torch.gather(probs, 1, action)

        else:
            select_probs, action = torch.topk(probs, topN, dim=1)

        entropy = - torch.sum(probs * (probs).log(),dim=1)
        return select_probs, action, entropy


