'''BM25 class unsupervised'''
from collections import Counter
import math,json,tqdm


class BM25:
    def __init__(self,df_addr,avg_doc_len=11.9224,num_of_docs=442419):
        self.doc_freq_dict = {}
        self.avg_doc_len = avg_doc_len
        self.num_of_docs = num_of_docs
        # parameters for computation of bm25
        self.k1 = 1.2
        self.k2 = 100
        self.b = 0.75
        self.data_init(df_addr)

    def data_init(self,addr):
        print('loading doc freq...')
        for line in open(addr):
            w,freq = line.strip().split('\t')
            self.doc_freq_dict[w] = int(freq)

    def estimate_w_bm25(self,qw,qf,doc,df,d_freq_counter):
        avdl = float(len(doc)) / self.avg_doc_len
        k = self.k1 * ((1 - self.b) + self.b * avdl)
        dfc = d_freq_counter[qw] if qw in d_freq_counter else 0
        idf = math.log((self.num_of_docs - df + 0.5) / (df + 0.5))
        score = idf * (((self.k1 + 1) * dfc) / float(k + dfc)) * (((self.k2 + 1) * qf) / float(self.k2 + qf))
        return score

    def run_query(self,query,docs):
        '''
        :param query: raw text
        :param doc: raw text
        :return:
        '''
        doc_scores = []
        for doc in docs:
            query_counter = Counter(query)
            d_freq_counter = Counter(doc)
            doc_score = 0.
            for term in query_counter:
                df = self.doc_freq_dict[term] if term in self.doc_freq_dict else 0
                qf = query_counter[term]
                doc_score += self.estimate_w_bm25(term, qf, doc, df, d_freq_counter)
            doc_scores.append(doc_score)
        return doc_scores

def main():
    from collections import defaultdict
    import os,sys
    sys.path.append('../')
    from utils import *
    from metrics.rank_evaluations import *
    bm25_model = BM25('/deeperpool/lixs/KQSM/data/processing/bm25_df.txt')
    test_addr = '/deeperpool/lixs/sessionST/ad-hoc-udoc/valid'
    is_test = 'test' in test_addr
    eval_label = 'PSCM'
    #eval_label = 'HUMAN' if is_test else 'PSCM'

    test_files = os.listdir(test_addr)
    evaluator = rank_eval(rel_threshold=1)
    results = defaultdict(list)
    gt_set = defaultdict(lambda :0)
    for test_file in tqdm.tqdm(test_files):
        gts = []
        docs = []
        for i,line in enumerate(open(os.path.join(test_addr,test_file))):
            if i == 0:
                continue
            elements = line.strip().split('\t')
            query_terms = filter_title(elements[2].split())
            doc_terms = filter_title(elements[3].split())
            index = -7 if is_test else -6
            labels = elements[index:]
            label = float(labels[model2id(eval_label)])
            gts.append(label)#human label
            for gt in gts:
                gt_set[int(gt)] += 1
            docs.append(doc_terms)
        pred_scores = bm25_model.run_query(query_terms,docs)
        #print('gts: ', gts)
        #print('pred_scores: ', pred_scores)
        result = evaluator.eval(gts,pred_scores,label=eval_label)

        for k,v in result.items():
            results[k].append(v)

    performances = {}
    for k, v in results.items():
        performances[k] = np.mean(v)
    print('----------------BM25 performance---------------------')
    print_performance(performances, 'BM25 performance: ' + eval_label)

    for k,v in gt_set.items():
        print ('Label %d\tRatio: %.2f' % (k,float(v)/sum(gt_set.values())))
if __name__ == '__main__':
    main()






