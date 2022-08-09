#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os,json,tqdm,cPickle
import numpy as np
data_addr = '/deeperpool/lixs/sessionST/ad-hoc-udoc/'
data_folders = ['train','valid','test']

df = {}
avg_doc_lens = []
for data_folder in data_folders:
    addr = os.path.join(data_addr,data_folder)
    qfiles = os.listdir(addr)
    for qfile in tqdm.tqdm(qfiles):
        file_addr = os.path.join(addr, qfile)
        for i,line in enumerate(open(file_addr)):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            query_terms = elements[2]
            doc_terms = elements[3].split()
            for w in doc_terms:
                if w in df:
                    df[w] += 1
                else:
                    df[w] = 1

            avg_doc_lens.append(len(doc_terms))

avg_doc_len = np.mean(avg_doc_lens)
num_of_docs = len(avg_doc_lens)

print 'avg_doc_len: ', avg_doc_len
print 'num_of_docs: ', num_of_docs

output_file = open('./bm25_df.txt','w')
for w in df:
    print >> output_file, w + '\t'+ str(df[w])
'''
output_json = {'avg_doc_len':avg_doc_len, 'num_of_docs':num_of_docs, 'df': df}
output_file = open('./bm25_df.json','w')
json.dump(output_json, output_file,ensure_ascii=False)
'''




