import argparse


def load_arguments():
    parser = argparse.ArgumentParser(description='Evidence Reading Model')

    parser.add_argument('--resume', type=str, default="",
                        help='Resume training from that state')
    parser.add_argument("--prototype", type=str, help="Use the prototype", default='basic_config')
    parser.add_argument("--eval", action="store_true", help="only evaluation")
    parser.add_argument('--ranker', type=str, default="BM25",
                        help='Ranker model in RL')
    parser.add_argument('--m', type=str, default="",
                        help='Message in visualizationn window (Flag)')
    parser.add_argument('--gpu', type=int, default=1,
                        help="# of GPU running on")
    parser.add_argument("--train_ranker", action="store_true", help="training the ranker")
    parser.add_argument("--use_knowledge", action="store_false", help="use entity input and candidates")
    parser.add_argument("--wzj",action="store_true",help="use the description data")
    args = parser.parse_args()

    return args

def basic_config():
    state = {}
    state['min_score_diff'] = 0.25

    #original_addr = '/ivi/ilps/personal/xli/sessionST'
    original_addr = '/deeperpool/lixs/sessionST'

    state['train_addr'] = original_addr + '/ad-hoc-udoc/train/'
    state['valid_addr'] = original_addr + '/ad-hoc-udoc/valid/'
    state['test_addr'] = original_addr + '/ad-hoc-udoc/test/'

    state['vocab_dict_file'] = original_addr + '/GraRanker/data/vocab.dict.9W.pkl'

    state['word_emb'] = original_addr + '/GraRanker/data/emb50_9W.pkl'
    #state['ent_emb'] = '../baselines/data/ent_embedding_50.pkl'
    state['saveModeladdr'] = './trainModel/'
    state['visual_path'] = './visual/'

    #state['ent_cand_term_addr'] = '../KQSM-1222/data/bert_process/qid2eterm_top100.pkl'
    state['ent_cand_term_addr_origin'] = '../KQSM-1222/data/processing/qid_eterm_top_sim100.pkl'
    state['ent_cand_term_addr_wzj'] = './data/processing/qid2know_terms_top100_wzj.json'

    state['ent_emb_addr'] = '../KQSM-1222/data/processing/ent_embedding_50.pkl'
    state['ent_dict_file'] = '../KQSM-1222/data/processing/ent2id_dict.pkl'
    #state['qid2eid_dict_file'] = './data/processing/parsed_qid2eid.xlore-ST.pkl' #entity id in query
    state['qid2eidx_file'] = '/deeperpool/lixs/knowledge/statistic/entity/RL-folder/qid2eidx.pkl'

    state['drate'] = 0.8 #dropout rate
    state['random_seed'] = 1234

    state['weight_decay'] = 0#1e-3

    state['clip_grad'] = 0.5
    state['optim'] = 'adam'  # 'sgd, adadelta*' adadelta0.1, adam0.005,adagrad

    state['word_emb_size'] = 50
    state['ent_emb_size'] = 50


    state['cost_threshold'] = 0.997
    state['patience'] = 5

    state['lr'] = 1e-5  # 0.01

    state['num_epochs'] = 200
    state['batch_size'] = 50 #50
    state['train_freq'] = 50 #50

    state['value_loss_alpha'] = 0.5
    state['random_mode'] = False
    state['mean_baseline'] = False#True
    return state

def term_config():
    state = basic_config()
    state['kernel_size'] = 3
    state['max_query_length'] = 12
    state['max_query_ent_len'] = 30 #12 + 15
    state['max_doc_length'] = 15

    state['mean_baseline'] = False
    state['q_embed_size'] = 128
    state['pool_size_q'] = state['max_query_ent_len'] - state['kernel_size'] + 1
    #state['pool_size_d'] = state['max_doc_length'] - state['kernel_size'] + 1

    state['cand_term_addr'] = '/deeperpool/lixs/KQSM/data/processing/candidate/bm25_top5_doc.term.json'
    state['bm25_df'] = '/deeperpool/lixs/KQSM/data/processing/bm25_df.txt'
    state['rewards'] = ['map']#,'ndcg@3'
    state['rel_threshold'] = 0

    state['num_repeat'] = 1#5
    state['num_epochs'] = 1000#5000
    state['lr'] = 1e-5
    state['weight_decay'] = 1e-4
    state['sample_size'] = 1
    state['patience'] = 500#TODO
    state['optim'] = 'adadelta'  # 'sgd, adadelta*' adadelta0.1, adam0.005,adagrad
    #state['optim'] = 'adadelta'
    state['use_scheduler'] = True#True
    state['mean_baseline'] = False  #False
    state['entropy_coef'] = 0.001#0.001#0.2
    return state

def test_config():
    state = term_config()
    state['num_repeat'] = 1
    #state['batch_size'] = 1
    #state['train_freq'] = 1
    state['num_epochs'] = 2000
    state['rewards'] = ['map']
    state['lr'] = 1e-2#2
    state['optim'] = 'adadelta'
    state['sample_size'] = 3
    state['patience'] = 10
    return state

def knrm_config():
    state = term_config()
    state['knrm_model_addr'] = '../KQSM-1222/data/knrm_process/knrm_model_20201226_fixnorm.pth'
    state['max_query_length'] = 10#12
    state['max_doc_length'] = 15#20
    state['max_query_ent_len'] = 30  # TODO
    state['pool_size_q'] = state['max_query_ent_len'] - state['kernel_size'] + 1

    state['max_expand_query_length'] = 15#20
    state['mean_baseline'] = False
    state['num_epochs'] = 1000#200
    state['num_repeat'] = 1#5
    state['batch_size'] = 50
    state['sample_size'] = 3
    state['lr'] = 1e-3#4
    state['patience'] = 300

    state['embsize'] = state['word_emb_size']
    state['kernel_size'] = 3
    state['out_size'] = 32

    state['kernel_num'] = 11
    state['sigma'] = 0.1
    state['exact_sigma'] = 0.001
    state['optim'] = 'adam'  # 'sgd, adadelta*' adadelta0.1, adam0.005,adagrad

    state['ranker_bs'] = 300
    state['ranker_optim'] = 'adam'
    state['ranker_lr'] = 1e-4
    state['train_ranker_freq'] = 10
    return state

def bert_config():
    state = term_config()
    state['bert_model_addr'] = '../KQSM-1222/data/bert_process/bert_model_full_maxL50.pth' #bert_model_full.pth,bert_model_full_maxL50.pth
    state['BERT_folder'] = '/deeperpool/lixs/sessionST/baselines/bert/bert-base-chinese'  # your path for model and vocab
    state['BERT_VOCAB'] = 'bert-base-chinese-vocab.txt'
    state['text_dim'] = 768
    state['output_layer_index'] = -1
    state['max_concat_text_length'] = 50#50
    state['optim'] = 'adam'
    state['rel_threshold'] = 0

    state['mean_baseline'] = False
    state['num_epochs'] = 500
    state['num_repeat'] = 1
    state['batch_size'] = 50
    state['mean_baseline'] = False
    state['sample_size'] = 3#3
    state['random_mode'] = False

    state['lr'] = 1e-3  # 4
    state['ranker_bs'] = 50#80
    state['ranker_optim'] = 'adam'
    state['ranker_lr'] = 3e-6#1e-6#3e-6
    state['train_ranker_freq'] = 20#5
    #state['batch_size'] = 10
    #state['lr'] = 3e-6
    #state['out_lr'] = 0.1
    #state['optim'] = 'adam'
    return state

