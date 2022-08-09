# encoding=utf8
import numpy as np
import cPickle,re
import random,time
import tqdm,os,json
from collections import defaultdict
from utils import *

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
    #print 'model_name: ',model_name
    models = ['TACM','PSCM','THCM','UBM','DBN','POM','HUMAN']
    return models.index(model_name)

class RankerDataGenerator():
    def __init__(self, config, word2id, qid2cand_terms,use_knowledge=True,is_descrption=False):
        #super(DataGenerator, self).__init__(config)
        self.config = config
        self.word2id = word2id
        self.vocab_size = len(word2id)
        self.train_rank_addr = config['train_addr']
        self.qid2cand_terms = qid2cand_terms
        self.qid2eidx = cPickle.load(open(config['qid2eidx_file']))
        #self.qid2cand_eterms = cPickle.load(open(config['ent_cand_term_addr']))
        self.use_knowledge = use_knowledge

        #for qid in self.qid2cand_eterms:
        #    self.qid2cand_eterms[qid] = map(lambda w: w.encode('utf8'), self.qid2cand_eterms[qid])

        qid2cand_eterms_addr = config['ent_cand_term_addr']
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
                    self.qid2cand_eterms[qid] = data['cand_terms']

    def ranking_pair_reader(self,batch_size):
        click_model = 'PSCM'
        model_id = model2id(click_model)
        text_list = os.listdir(self.train_rank_addr)

        query_batch, doc_pos_batch, doc_neg_batch, qid_batch, candidate_batch = [],[],[],[],[]
        query_ent_batch = []
        max_q_len = self.config['max_query_length']
        max_qe_len = self.config['max_query_ent_len']
        max_d_len = self.config['max_doc_length']

        max_seq_len = max(max_d_len,max_q_len)
        random.shuffle(text_list)

        for text_id in text_list:
            documents = [];relevances = [];qids = []
            for i,line in enumerate(open(os.path.join(self.train_rank_addr,text_id))):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                qid = elements[0]
                query = elements[2]

                title_idx = map(lambda w: find_id(self.word2id, w), filter_title(elements[3].split()))[:max_seq_len]

                if len(title_idx) == 0:
                    continue
                documents.append(title_idx)

                labels = map(float,elements[-6:])
                relevances.append(labels[model_id])

            query_idx = map(lambda w: find_id(self.word2id, w),filter_title(query.split()))[:max_seq_len]
            query_ent_idx = query_idx + map(lambda t: t + self.vocab_size, self.qid2eidx[qid]) if qid in self.qid2eidx \
                                                                                                  and self.use_knowledge else query_idx
            cand_terms = self.qid2cand_terms[qid]
            if self.use_knowledge and qid in self.qid2cand_eterms:
                set_terms = set(cand_terms)
                set_terms.update(self.qid2cand_eterms[qid])
                cand_terms = list(set_terms)
            cand_terms = map(lambda w: find_id(self.word2id, w), cand_terms)

            for i in range(len(documents) - 1):
                for j in range(i + 1, len(documents)):
                    pos_i,neg_i = i, j
                    y_diff = relevances[pos_i] - relevances[neg_i]
                    if abs(y_diff) < self.config['min_score_diff']:
                        continue
                    if y_diff < 0:
                        pos_i, neg_i = neg_i, pos_i

                    pos_doc = documents[pos_i]
                    neg_doc = documents[neg_i]

                    qid_batch.append(qid)
                    candidate_batch.append(cand_terms)
                    query_batch.append(query_idx)
                    query_ent_batch.append(query_ent_idx)
                    doc_pos_batch.append(pos_doc)
                    doc_neg_batch.append(neg_doc)

                    if len(query_batch) >= batch_size:
                        query_batch = np.array([self.pad_seq(s[:max_q_len], max_q_len) for s in query_batch])
                        query_ent_batch = np.array([self.pad_seq(s[:max_qe_len], max_qe_len) for s in query_ent_batch])
                        doc_pos_batch = np.array([self.pad_seq(d[:max_d_len], max_d_len) for d in doc_pos_batch])
                        doc_neg_batch = np.array([self.pad_seq(d[:max_d_len], max_d_len) for d in doc_neg_batch])
                        max_cand_len = np.max(map(lambda d:len(d), candidate_batch))
                        candidate_batch = np.array([self.pad_seq(c[:max_cand_len], max_cand_len) for c in candidate_batch])
                        yield (query_batch,query_ent_batch, doc_pos_batch, doc_neg_batch, candidate_batch)
                        query_batch, doc_pos_batch, doc_neg_batch, qid_batch, candidate_batch = [],[],[],[],[]
                        query_ent_batch = []


    def pad_seq(self, seq, max_length, PAD_token=0):
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

class BERTDataGenerator():
    def __init__(self, config, word2id, qid2cand_terms,use_knowledge=True,is_descrption=False):
        # super(DataGenerator, self).__init__(config)
        self.config = config
        self.word2id = word2id
        self.vocab_size = len(word2id)
        self.train_rank_addr = config['train_addr']
        self.qid2cand_terms = qid2cand_terms
        self.qid2eidx = cPickle.load(open(config['qid2eidx_file']))
        self.max_query_length = config['max_query_length']
        self.use_knowledge = use_knowledge

        qid2cand_eterms_addr = config['ent_cand_term_addr']
        if not is_descrption:
            self.qid2cand_eterms = cPickle.load(open(qid2cand_eterms_addr))
            for qid in self.qid2cand_eterms:
                self.qid2cand_eterms[qid] = map(lambda w: w.encode('utf8'), self.qid2cand_eterms[qid])
        else:
            # from zhijing's knowledge file
            self.qid2cand_eterms = {}
            qid2know_terms = json.load(open(qid2cand_eterms_addr))
            for qid, data in qid2know_terms.iteritems():
                if len(data['cand_terms']) > 0:
                    self.qid2cand_eterms[qid] = data['cand_terms']

    def bert_pair_reader(self, batch_size):

        click_model = 'PSCM'
        model_id = model2id(click_model)
        text_list = os.listdir(self.config['train_addr'])

        query_batch, pos_doc_batch, neg_doc_batch,raw_query_text,candidate_batch,query_ent_batch = [],[],[],[],[],[]

        random.shuffle(text_list)
        max_qe_len = self.config['max_query_ent_len']

        for text_name in text_list:
            filepath = os.path.join(self.config['train_addr'], text_name)
            relevances = [];
            documents = []
            for i, line in enumerate(open(filepath)):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                qid = elements[0]
                query_terms = elements[2]
                doc_content = elements[3]

                if len(doc_content.split()) == 0:
                    continue

                labels = map(float, elements[-6:])
                relevances.append(labels[model_id])
                documents.append(doc_content)

            if len(query_terms.split()) == 0:
                continue
            query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query_terms.split()))[:self.max_query_length]
            #cand_terms = map(lambda w: find_id(self.word2id, w), self.qid2cand_terms[qid])

            #query_idx = map(lambda w: find_id(self.word2id, w), filter_title(query.split()))[:max_seq_len]
            query_ent_idx = query_idx + map(lambda t: t + self.vocab_size, self.qid2eidx[qid]) if qid in self.qid2eidx \
                                                                                                  and self.use_knowledge else query_idx
            cand_terms = self.qid2cand_terms[qid]
            if self.use_knowledge and qid in self.qid2cand_eterms:
                set_terms = set(cand_terms)
                set_terms.update(self.qid2cand_eterms[qid])
                cand_terms = list(set_terms)
            cand_terms = map(lambda w: find_id(self.word2id, w), cand_terms)

            docs_size = len(documents)
            for i in range(docs_size - 1):
                for j in range(i, docs_size):
                    pos_i, neg_i = i, j
                    y_diff = relevances[pos_i] - relevances[neg_i]
                    if abs(y_diff) < self.config['min_score_diff']:
                        continue
                    if y_diff < 0:
                        pos_i, neg_i = neg_i, pos_i

                    pos_doc = documents[pos_i]
                    neg_doc = documents[neg_i]

                    query_ent_batch.append(query_ent_idx)
                    raw_query_text.append(query_terms)
                    query_batch.append(query_idx)
                    pos_doc_batch.append(pos_doc)
                    neg_doc_batch.append(neg_doc)
                    candidate_batch.append(cand_terms)
                    if len(query_batch) >= batch_size:
                        query_batch = np.array([self.pad_seq(s, self.max_query_length) for s in query_batch])
                        query_ent_batch = np.array([self.pad_seq(s, max_qe_len) for s in query_ent_batch])
                        max_cand_len = np.max(map(lambda d: len(d), candidate_batch))
                        candidate_batch = np.array(
                            [self.pad_seq(c, max_cand_len) for c in candidate_batch])

                        yield query_batch,raw_query_text,query_ent_batch,candidate_batch, pos_doc_batch, neg_doc_batch
                        query_batch, pos_doc_batch, neg_doc_batch, raw_query_text, candidate_batch, query_ent_batch = [], [], [], [], [], []

    def pad_seq(self, seq, max_length, PAD_token=0):
        seq = seq[:max_length]
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

    def pad_seq_bert(self, seq, max_length, PAD_token=103):
        # id of [MASK] is 103
        if len(seq) > max_length:
            seq = seq[:max_length - 1] + [seq[-1]]
        seq += [PAD_token for i in range(max_length - len(seq))]
        return seq

