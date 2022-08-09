# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import cPickle,os,json

max_query_length = 12#10
max_doc_length = 15#120

useless_words = ['-','——','_','【','】','(',')','.',',','《','》','?','、','（','）','。',':','，','・']

def filter_title(doc_words):
    words = []
    for w in doc_words:
        if len(w) == 0 or w in useless_words:
            continue
        words.append(w)
    return words

def find_id(word_dict,word):
    return word_dict[word] if word in word_dict else 1

def model2id(model_name):
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)

class CommonResource():
    def __init__(self,cand_term_addr,qid2eidx_addr,qid2cand_eterms_addr,word2id,use_knowledge=True,is_descrption=False):
        self.qid2query, self.qid2cand_terms = self.load_query_info(cand_term_addr)
        self.qid2eidx = cPickle.load(open(qid2eidx_addr))

        if not is_descrption:
            self.qid2cand_eterms = cPickle.load(open(qid2cand_eterms_addr))
            for qid in self.qid2cand_eterms:
                self.qid2cand_eterms[qid] = map(lambda w: w.encode('utf8'), self.qid2cand_eterms[qid])
        else:
            #from zhijing's knowledge file
            self.qid2cand_eterms = {}
            qid2know_terms = json.load(open(qid2cand_eterms_addr))
            for qid,data in qid2know_terms.iteritems():
                if len(data['cand_terms']) > 0:
                    self.qid2cand_eterms[qid] = map(lambda w: w.encode('utf8'), data['cand_terms'])

        #print('word2id: ', word2id.keys()[:2])
        #print('qid2cand_terms： ',self.qid2cand_terms.values()[:2])
        #print('qid2cand_eterms： ', self.qid2cand_eterms.values()[:2])
        #exit()
        self.word2id = word2id
        self.vocab_size = len(word2id)
        self.use_knowledge = use_knowledge

    def load_query_info(self, addr):
        qid2cand_terms = {}
        qid2query = {}
        query_json_list = json.load(open(addr))
        for data in query_json_list:
            qid = data['qid']
            query = data['query']
            cand_terms = data['cand_terms']
            qid2query[qid] = str(query)
            qid2cand_terms[qid] = map(lambda term: str(term), cand_terms)
        return qid2query, qid2cand_terms

class EntityDataset(Dataset):
    """Class to load Query dataset."""
    def __init__(self,data_addr,common_resource,ground_truth='PSCM'):
        self.data_addr = data_addr
        self.qfile_list = os.listdir(data_addr)#[:2000]
        self.qid2query = common_resource.qid2query
        self.qid2cand_terms = common_resource.qid2cand_terms
        self.qid2cand_eterms = common_resource.qid2cand_eterms
        self.word2id = common_resource.word2id
        self.qid2eidx = common_resource.qid2eidx
        self.use_knowledge = common_resource.use_knowledge
        self.vocab_size = common_resource.vocab_size

        self.is_test = 'test' in data_addr
        if not self.is_test:
            self.qfile_list = self.qfile_list[:]
        self.ground_truth_idx = model2id(ground_truth)

    def __len__(self):
        """Return length of dataset."""
        return len(self.qfile_list)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        qfile = self.qfile_list[i]
        filepath = os.path.join(self.data_addr, qfile)

        docs = []
        gts = []
        raw_docs = []
        for i, line in enumerate(open(filepath)):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            qid, docid = elements[:2]
            query_terms = elements[2]
            doc_content = elements[3]
            if len(doc_content.split()) == 0:
                continue
            doc_idx = map(lambda w: find_id(self.word2id, w), filter_title(elements[3].split()))[:max_doc_length]
            index = -7 if self.is_test else -6
            labels = map(float, elements[index:])
            raw_docs.append(elements[3])
            gts.append(labels[self.ground_truth_idx])
            docs.append(doc_idx)
        query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query_terms.split()))[
                    :max_query_length]

        query_ent_idx = query_idx + map(lambda t:t + self.vocab_size,self.qid2eidx[qid]) if qid in self.qid2eidx \
                                                                                            and self.use_knowledge else query_idx
        raw_query_text = query_terms
        cand_terms = self.qid2cand_terms[qid]#[:5]
        #print('cand_terms:', len(cand_terms))

        #print('cand_terms: ', cand_terms)
        if self.use_knowledge and qid in self.qid2cand_eterms:
            set_terms = set(cand_terms)
            set_terms.update(self.qid2cand_eterms[qid])
            cand_terms = list(set_terms)
        #print('Total terms:', len(cand_terms))
        #cand_terms = cand_terms#[:10]
        #query_cand_terms = map(lambda w:find_id(self.word2id,w), cand_terms)
        #print('cand_terms: ', cand_terms)
        #exit()
        query_cand_terms = []
        for w in cand_terms:
            idx = find_id(self.word2id, w)
            if idx > 1:
                query_cand_terms.append(idx)

        # package data to FloatTensor or cuda()
        #print('==================================')
        #print('query_idx:',query_idx)
        #print('query_ent_idx:',query_ent_idx)
        #print('raw_query_text:',raw_query_text)
        #print('query_cand_terms:',query_cand_terms)
        #exit(-1)
        return {'qid':qid,'query': query_idx,'query_ent_idx':query_ent_idx,'raw_query_text':raw_query_text,'raw_docs':raw_docs,\
                'docs':docs, 'query_cand_terms':query_cand_terms, 'gts':gts}


def main_test():
    import sys
    sys.path.append('../')
    from utils import *
    vocab_dict_file = '/deeperpool/lixs/sessionST/GraRanker/data/vocab.dict.9W.pkl'
    word2id, id2word = cPickle.load(open(vocab_dict_file))
    print('word id dict loaded...')
    train_dataset = TermDataset(data_addr='/deeperpool/lixs/sessionST/ad-hoc-udoc/train',\
                                cand_term_addr='./processing/candidate/bm25_top3_doc.term.json',word2id=word2id)
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=False,collate_fn=truncate)

    for batch_samples in train_loader:
        #query_batch = batch_samples['query']
        #docs_batch = batch_samples['docs']
        #print query_batch
        print('--------------------------------------')
        print(batch_samples)
        #print docs_batch
        exit(-1)

if __name__ == '__main__':
    #for testing
    main_test()


