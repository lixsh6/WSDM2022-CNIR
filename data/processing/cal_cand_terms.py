#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

'''
calculate candidate terms for each query ID
Methods:
    1. terms from neigborhoods in KG
    2. terms from all retrieved documents
'''


import os,json,tqdm,cPickle
import numpy as np
from collections import defaultdict
data_addr = '/deeperpool/lixs/sessionST/ad-hoc-udoc/'
data_folders = ['train','valid','test']

useless_words = ['-','——','_','【','】','(',')','.',',','《','》','?','、','（','）','。',':','，','・','-','|','/','？','的']

def filter_title(doc_words):
    words = []
    for w in doc_words:
        if len(w) == 0 or w in useless_words:
            continue
        words.append(w)
    return words

df = {}
avg_doc_lens = []

qid2candset = defaultdict(set)
qid2query = {}

save_file_name = ''
def from_all_retrieved_docs():
    for data_folder in data_folders:
        addr = os.path.join(data_addr,data_folder)
        qfiles = os.listdir(addr)
        for qfile in tqdm.tqdm(qfiles):
            file_addr = os.path.join(addr, qfile)
            for i,line in enumerate(open(file_addr)):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                qid = elements[0]
                did = elements[1]
                query_terms = elements[2]
                doc_terms = filter_title(elements[3].split())

                qid2candset[qid].update(doc_terms)
                qid2query[qid] = query_terms
    global save_file_name
    save_file_name = 'all_retrieved_doc.term.json'

def from_top_bm25_docs(topK):
    import sys
    sys.path.append('../../')
    from ranker.bm25 import BM25
    import itertools
    bm25_model = BM25('/deeperpool/lixs/KQSM/data/processing/bm25_df.txt')

    for data_folder in data_folders:
        addr = os.path.join(data_addr,data_folder)
        qfiles = os.listdir(addr)
        for qfile in tqdm.tqdm(qfiles):
            file_addr = os.path.join(addr, qfile)
            docs = []
            pred_scores = []
            for i,line in enumerate(open(file_addr)):
                if i == 0:
                    continue
                elements = line.strip().split('\t')
                qid = elements[0]
                did = elements[1]
                query_terms = elements[2]
                raw_doc_terms = elements[3].split()
                doc_terms = filter_title(raw_doc_terms)
                docs.append(doc_terms)
                pred_scores.append(bm25_model.run_query(query_terms, raw_doc_terms))
            top_doc_terms = [doc for _, doc in sorted(zip(pred_scores, docs), key=lambda pair: pair[0], reverse = True)[:topK]]
            terms = list(itertools.chain(*top_doc_terms))
            qid2candset[qid].update(terms)
            qid2query[qid] = query_terms
    global save_file_name
    save_file_name = 'bm25_top%d_doc.term.json' % topK


##################################Key function to execute#################################################
from_all_retrieved_docs()
#from_top_bm25_docs(3)

#########################################################################################################


query_cand_term_json = []
cand_lengths = []
for qid in qid2candset:
    query_terms = qid2query[qid]
    cand_terms = list(qid2candset[qid])
    cand_lengths.append(len(cand_terms))
    data = {'qid':qid,'query':query_terms,'cand_terms':cand_terms}
    query_cand_term_json.append(data)

print 'Query Number: ', len(query_cand_term_json)
print 'Avg Candidate Number: ', np.mean(cand_lengths)
#print query_cand_term_json
if not os.path.exists('./candidate/'):
    os.makedirs('./candidate/')

print 'save_file_name: ', save_file_name
json.dump(query_cand_term_json, open('./candidate/' + save_file_name,'w'), ensure_ascii=False)





