# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import cPickle,tqdm,os
from collections import defaultdict
ent_embed_file = '/deeperpool/lixs/sessionST/baselines/LINE/ent/xlore_50_line.txt'

#ent_addr = '/deeperpool/lixs/knowledge/statistic/entity/data/raw-xlore.ent_dict.pkl'
ent_addr = '/deeperpool/lixs/knowledge/statistic/entity/v2/xlore_data/raw-xlore.ent_dict-multiName.pkl'
print('loading...')

qd2id_dict_file = '/deeperpool/lixs/knowledge/statistic/entity/RL-folder/qid2eid.xlore-ST.pkl'
qid2eid = cPickle.load(open(qd2id_dict_file))#dict: qid(str) -> list

qd2id_dict_file = '/deeperpool/lixs/knowledge_intent/data/qdid2eid-xlore.pkl'
qid2eid_old, did2eid = cPickle.load(open(qd2id_dict_file))

print(len(qid2eid),len(qid2eid_old))

def get_ent_set(qid2eid):
    ent_set = set()
    for qid in qid2eid:
        ent_list = qid2eid[qid]
        ent_set.update(map(lambda s:s.strip(),ent_list))
    return ent_set

ent_set = get_ent_set(qid2eid)
ent_set_old = get_ent_set(qid2eid_old)
print(len(ent_set),len(ent_set_old))
c = 0
for eid in ent_set_old:
    if eid not in ent_set:
        c += 1
#print c
#print ent_set_old
#print ent_set
#exit(-1)
(ent2id, id2ent) = cPickle.load(open(ent_addr))
print('ent2id: ', len(ent2id),len(id2ent))
#

embed_dict = {}
for i,line in tqdm.tqdm(enumerate(open(ent_embed_file))):
    if i > 0:
        elements = line.strip().split()
        #if elements[0].strip() in ent_set:
        embed_dict[elements[0].strip()] = map(float,elements[1:])

print('embed_dict: ', len(embed_dict))
ent2idx,idx2ent = {'<m>':0,'<unk>':1},{'<m>':0,'<unk>':1}
ent_embedding = np.random.random((len(ent_set) + 2,50)) #mask, unk, etc...

find_q = 0
for entid in tqdm.tqdm(ent_set):
    idx = len(ent2idx)
    ent = id2ent[entid]
    ent2idx[ent] = idx
    idx2ent[idx] = ent

    found = 0
    for eid in ent2id[ent]:
        if eid in embed_dict:
            ent_embedding[idx] = embed_dict[eid]
            found = 1
    find_q += found


new_qid2eid = {}
len_dict = defaultdict(lambda : 0)
for qid,ent_list in qid2eid.iteritems():
    new_list = []
    for entid in ent_list:
        ent = id2ent[entid]
        if ent in ent2idx:
            new_list.append(ent2idx[ent])
    new_qid2eid[qid] = new_list
    len_dict[len(new_list)] += 1

print('found: ',find_q)
print('size: ', ent_embedding.shape)
print 'len: ', len_dict

cPickle.dump((ent2idx,idx2ent), open('./ent2id_dict.pkl','w'))
cPickle.dump(ent_embedding, open('./ent_embedding_50.pkl','w'))
cPickle.dump(new_qid2eid, open('./parsed_qid2eid.xlore-ST.pkl','w'))







