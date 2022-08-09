#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import shutil,os
from tensorboardX import SummaryWriter
use_cuda = torch.cuda.is_available() #and False
from pytorch_pretrained_bert import BertTokenizer, BertModel

def Tensor2Varible(tensor_,requires_grad = False):
    var = Variable(tensor_,requires_grad = requires_grad)
    var = var.cuda() if use_cuda else var
    return var

#FOR BERT

def pad_seq_bert(seq, max_length, PAD_token=103):
    # id of [MASK] in BERT is 103
    if len(seq) > max_length:
        seq = seq[:max_length - 1] + [seq[-1]]
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def word_split(text):
    text = unicode(text, 'utf-8')
    return [i.strip() for i in text]

def bert_convert_ids(text_input1,text_input2,tokenizer):
    text_list = ['[CLS]'] + word_split(text_input1) + ['[SEP]']+ word_split(text_input2) + ['[SEP]']
    text = ' '.join(text_list).decode('utf8')
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens

class CNN1D(torch.nn.Module):
    def __init__(self,in_dim=50,out_dim=32,k_size=3,pool_size=2):
        super(CNN1D, self).__init__()
        self.convs = nn.Conv1d(in_dim, out_dim, k_size) #batch_size x out_dim*(len - k_size + 1)
        self.pool_layer = nn.MaxPool1d(pool_size)
        self.out_layer = nn.Sigmoid()
    def forward(self,input):
        '''
        :param input:(b_s,textlen,embsize)
        :return:(b_s,out_dim)
        '''
        text = input.permute(0, 2, 1)
        x = self.convs(text)
        x2 = self.pool_layer(x)
        x2 = x2.squeeze(-1)
        x2 = self.out_layer(x2)
        return x2

def get_raw_text(raw_query_batch,cand_batch,id2word):
    new_query_batch = []
    local_batch_size = len(raw_query_batch)#len(cand_batch) / num_repeat
    for i in range(len(cand_batch)):
        query_index = i % local_batch_size
        new_query_idx = list(raw_query_batch[query_index]) + list(cand_batch[i])
        new_query = []
        for idx in new_query_idx:
            #remove the padding token
            if idx != 0 and idx in id2word:
                new_query.append(idx)
        new_query_batch.append(new_query)
    return new_query_batch

def get_raw_text_bert(raw_query_text,cand_batch,id2word):
    new_query_batch = []
    local_batch_size = len(raw_query_text)#len(cand_batch) / num_repeat
    for i in range(len(cand_batch)):
        query_index = i % local_batch_size
        new_query = raw_query_text[query_index]

        for cand_idx in cand_batch[i]:
            if cand_idx != 0 and cand_idx in id2word:
                new_query += id2word[cand_idx]

        new_query_batch.append(new_query)
    return new_query_batch

def print_text_id(raw_query_batch_id,id2word):
    print('==============================')
    for raw_query_id in raw_query_batch_id:
        text = ''
        for idx in raw_query_id:
            if idx != 0 and idx in id2word:
                text += ' ' + id2word[idx]
        print(text)
    print('==============================')

def print_text_comparison(raw_query_batch,new_query_batch,rewards, id2word):
    print('==============================')
    for i,(raw_query,new_query) in enumerate(zip(raw_query_batch,new_query_batch)):
        if rewards[i] <= 0:
            continue

        raw = ''
        for idx in raw_query:
            if idx != 0 and idx in id2word:
                raw += ' ' + id2word[idx]
        new = ''
        for idx in new_query:
            if idx != 0 and idx in id2word:
                new += ' ' + id2word[idx]
        print(raw)
        print(new)
        print('rewards: ', rewards[i])
        if np.isnan(rewards[i]):
            exit(-1)
    print('==============================')


def visualizer(filewriter_path,message, clear_old=True):
    filewriter_path = os.path.join(filewriter_path, message)
    #if clear_old and len(message) > 0 and os.path.exists(filewriter_path):   #delete the old file
    #    shutil.rmtree(filewriter_path)
    if not os.path.exists(filewriter_path):
        os.makedirs(filewriter_path)
    writer = SummaryWriter(filewriter_path, comment='visualizer')
    return writer

useless_words = ['-','——','_','【','】','(',')','.',',','《','》','?','、','（','）','。',':','，','・']

def filter_title(doc_words):
    words = []
    for w in doc_words:
        if len(w) == 0 or w in useless_words:
            continue
        words.append(w)
    return words

class MyCollator(object):
    def __init__(self, max_query_len,sample_size=0):
        self.max_query_len = max_query_len
        self.sample_size = sample_size
    def __call__(self, data_list):
        """传进一个batch_size大小的数据"""
        #max_query_len = max([len(data['query']) for data in data_list])
        #make sure every query is able to remain the original form
        max_cand_len = max([len(data['query_cand_terms']) for data in data_list]) + self.sample_size
        qid_batch = []
        raw_query_batch = []
        raw_query_text = []
        raw_doc_text = []
        query_batch = []
        cand_term_batch = []
        docs_batch = []
        gts_batch = []
        for data in data_list:
            qid_batch.append(data['qid'])
            raw_query_batch.append(data['query'])
            raw_query_text.append(data['raw_query_text'])
            raw_doc_text.append(data['raw_docs'])
            query_batch.append(np.array(pad_seq(data['query'], self.max_query_len)))
            cand_term_batch.append(np.array(pad_seq(data['query_cand_terms'],max_cand_len)))
            docs_batch.append(data['docs'])                                 #Don't pad it temporarily TODO
            gts_batch.append(data['gts'])
        return {'qid':qid_batch,'raw_query':raw_query_batch,'raw_query_text':raw_query_text,'raw_doc_text':raw_doc_text,\
                'query':query_batch,
                'query_cand_terms':cand_term_batch, 'docs':docs_batch, 'gts':gts_batch}

class RankerCollator(object):
    def __init__(self, max_query_len,sample_size=0):
        self.max_query_len = max_query_len
    def __call__(self, data_list):
        """传进一个batch_size大小的数据"""
        #max_query_len = max([len(data['query']) for data in data_list])
        #make sure every query is able to remain the original form
        max_cand_len = max([len(data['query_cand_terms']) for data in data_list]) + self.sample_size
        query_batch = []
        cand_term_batch = []
        docs_batch = []
        for data in data_list:
            qid_batch.append(data['qid'])
            raw_query_batch.append(data['query'])
            query_batch.append(np.array(pad_seq(data['query'], self.max_query_len)))
            cand_term_batch.append(np.array(pad_seq(data['query_cand_terms'],max_cand_len)))
            docs_batch.append(data['docs'])                                 #Don't pad it temporarily TODO
            gts_batch.append(data['gts'])
        return {'qid':qid_batch,'raw_query':raw_query_batch, 'query':query_batch,\
                'query_cand_terms':cand_term_batch, 'docs':docs_batch, 'gts':gts_batch}


class KnowledgeCollator(object):
    def __init__(self,max_query_len, max_query_ent_len,sample_size=0):
        self.max_query_len = max_query_len
        self.max_query_ent_len = max_query_ent_len
        self.sample_size = sample_size
    def __call__(self, data_list):
        """传进一个batch_size大小的数据"""
        #max_query_len = max([len(data['query']) for data in data_list])
        #make sure every query is able to remain the original form
        max_cand_len = max([len(data['query_cand_terms']) for data in data_list]) + self.sample_size
        qid_batch = []
        raw_query_batch = []
        raw_query_text = []
        raw_doc_text = []
        query_batch = []
        query_ent_idx_batch = []
        cand_term_batch = []
        docs_batch = []
        gts_batch = []
        for data in data_list:
            qid_batch.append(data['qid'])
            raw_query_batch.append(data['query'])
            raw_query_text.append(data['raw_query_text'])
            query_ent_idx_batch.append(np.array(pad_seq(data['query_ent_idx'], self.max_query_ent_len)))
            raw_doc_text.append(data['raw_docs'])
            query_batch.append(np.array(pad_seq(data['query'], self.max_query_len)))
            cand_term_batch.append(np.array(pad_seq(data['query_cand_terms'],max_cand_len)))
            docs_batch.append(data['docs'])                                 #Don't pad it temporarily TODO
            gts_batch.append(data['gts'])

        #print('max_cand_len:',max_cand_len)
        return {'qid':qid_batch,'raw_query':raw_query_batch,'raw_query_text':raw_query_text,'raw_doc_text':raw_doc_text,\
                'query':query_batch,'query_ent_idx_batch':query_ent_idx_batch,
                'query_cand_terms':cand_term_batch, 'docs':docs_batch, 'gts':gts_batch}

def pad_seq(seq, max_length, PAD_token=0):
    #print 'seq: ', seq,max_length
    if len(seq) > max_length:
        seq = seq[:max_length]
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def getOptimizer(name,parameters,**kwargs):
    if name == 'sgd':
        return optim.SGD(parameters,**kwargs)
    elif name == 'adadelta':
        return optim.Adadelta(parameters,**kwargs)
    elif name == 'adam':
        return optim.Adam(parameters,**kwargs)
    elif name == 'adagrad':
        return optim.Adagrad(parameters,**kwargs)
    elif name == 'rmsprop':
        return optim.RMSprop(parameters,**kwargs)
    else:
        raise Exception('Optimizer Name Error')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#==================================================
def print_performance(dict_,name=''):
    print('--------------%s---------------' % name)
    for metric in sorted(dict_.keys()):
        value = dict_[metric]
        print("{}:\t {:.4f}".format(metric, value))
    print('--------------%s---------------' % name)

def freeze_ranker(ranker,requires_grad=False):
    for p in ranker.parameters():
        p.requires_grad = requires_grad


class Attention(nn.Module):
    def __init__(self, hidden_size, h_state_embed_size=None, in_memory_embed_size=None, attn_type='simple'):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        if not h_state_embed_size:
            h_state_embed_size = hidden_size
        if not in_memory_embed_size:
            in_memory_embed_size = hidden_size
        if attn_type in ('mul', 'add'):
            self.W = torch.Tensor(h_state_embed_size, hidden_size)
            self.W = nn.Parameter(nn.init.xavier_uniform_(self.W))
            if attn_type == 'add':
                self.W2 = torch.Tensor(in_memory_embed_size, hidden_size)
                self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))
                self.W3 = torch.Tensor(hidden_size, 1)
                self.W3 = nn.Parameter(nn.init.xavier_uniform_(self.W3))
        elif attn_type == 'simple':
            pass
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))

    def forward(self, query_embed, in_memory_embed, attn_mask=None, addition_vec=None):
        if self.attn_type == 'simple': # simple attention
            attention = torch.bmm(in_memory_embed, query_embed.unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'mul': # multiplicative attention
            attention = torch.bmm(in_memory_embed, torch.mm(query_embed, self.W).unsqueeze(2)).squeeze(2)
            if addition_vec is not None:
                attention = attention + addition_vec
        elif self.attn_type == 'add': # additive attention
            attention = torch.mm(in_memory_embed.view(-1, in_memory_embed.size(-1)), self.W2)\
                .view(in_memory_embed.size(0), -1, self.W2.size(-1)) + torch.mm(query_embed, self.W).unsqueeze(1)
            if addition_vec is not None:
                attention = attention + addition_vec
            attention = torch.tanh(attention)
            attention = torch.mm(attention.view(-1, attention.size(-1)), self.W3).view(attention.size(0), -1)
        else:
            raise RuntimeError('Unknown attn_type: {}'.format(self.attn_type))

        if attn_mask is not None:
            # Exclude masked elements from the softmax
            attention = attn_mask * attention - (1 - attn_mask) * 1e20
        return attention