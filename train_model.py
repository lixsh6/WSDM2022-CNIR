    import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils import *
from collections import defaultdict
import datetime,os,math
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from model.knowledge_agent import *
from data.ent_dataset import *
from data.ranker_generator import *
class TRAIN_MODEL(object):
    def __init__(self,config,args):
        if not args.use_knowledge:
            config['max_query_ent_len'] = config['max_query_length']
            config['pool_size_q'] = config['max_query_length'] - config['kernel_size'] + 1
        if args.wzj:
            config['ent_cand_term_addr'] = config['ent_cand_term_addr_wzj']
        else:
            config['ent_cand_term_addr'] = config['ent_cand_term_addr_origin']
        self.config = config
        self.__dict__.update(config)


        self.word2id, self.id2word = cPickle.load(open(self.vocab_dict_file))
        self.ent2idx, self.idx2ent = cPickle.load(open(self.ent_dict_file))
        self.vocab_size = len(self.word2id)
        self.ent_size = len(self.ent2idx)
        print('Vocab size:', self.vocab_size)
        print('Entity size:', self.ent_size)
        print('Using Knowledge: ', args.use_knowledge)
        self.ranker_name = args.ranker
        self.train_ranker_bool = args.train_ranker
        self.use_knowledge = args.use_knowledge
        self.loc = 'cuda:{}'.format(args.gpu)

        self.select_agent = KnowledgeAgent(self.vocab_size, self.ent_size,self.use_knowledge, config)

        self.evaluator = rank_eval(rel_threshold=self.rel_threshold)
        if self.ranker_name == 'BM25':
            self.ranker = BM25(self.bm25_df)
        elif self.ranker_name == 'KNRM':
            self.ranker = KNRM(self.vocab_size, config)
            self.load_ranker(self.config['knrm_model_addr'], self.ranker)
            freeze_ranker(self.ranker)
        elif self.ranker_name == 'BERT':
            self.ranker = BERT(config)
            self.bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(config['BERT_folder'], config['BERT_VOCAB']))
            self.load_ranker(self.config['bert_model_addr'], self.ranker)
            freeze_ranker(self.ranker)
        print('==============Loaded==============')


        if args.resume and os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            self.load(args.resume)
        else:
            print("Creating a new model")

        agent_para_size = sum(p.numel() for p in self.select_agent.parameters())
        ranker_size = 0
        if use_cuda:
            torch.cuda.set_device(args.gpu)
            self.select_agent.cuda()
            if self.ranker_name != 'BM25':
                self.ranker.cuda()
                ranker_size = sum(p.numel() for p in self.ranker.parameters())
                print()
        print('Agent para: ', agent_para_size)
        print('Total size: ', agent_para_size + ranker_size)

        self.optimizer = torch.optim.Adagrad(self.select_agent.parameters(),
                                             lr=config['lr'], weight_decay=config['weight_decay'])
        if self.train_ranker_bool:
            self.ranker_optimizer = getOptimizer(self.ranker_optim, self.ranker.parameters(),
                                      lr=self.ranker_lr, betas=(0.99, 0.99))
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda f: 1 - f / self.num_epochs)

        self.is_descrption = 'wzj' in config['ent_cand_term_addr']
        self.common_resource = CommonResource(config['cand_term_addr'], config['qid2eidx_file'], \
                                         config['ent_cand_term_addr'],self.word2id,
                                              use_knowledge=self.use_knowledge,
                                              is_descrption=self.is_descrption
                                              )
        self.message = args.m
        # Using REINFORCE with a value baseline
        self.mse = nn.MSELoss()
        self.collate_fun = KnowledgeCollator(self.max_query_length,self.max_query_ent_len,self.sample_size)
        self.writer = visualizer(config['visual_path'], self.message)

    def load_ranker(self,ranker_addr,ranker):
        print("Ranker parameter size: ", count_parameters(ranker))
        print('Loading pretrained ranker: ', ranker_addr)
        model_state = torch.load(ranker_addr,map_location=self.loc)
        print('KEYS: ', model_state.keys())
        ranker.load_state_dict(model_state)

    def env_step(self,query_exp,docs_list,gts_batch,label='PSCM'):
        '''
        :param query_exp: expanded_query_batch, unpadded
        :param docs: doc_batch, unpadded
        :param gts: ground truth scores
        :return:
        '''
        reward_list = []
        metrics = defaultdict(list)
        local_batch_size = len(docs_list) #/ self.num_repeat

        for i,new_query in enumerate(query_exp):
            gts = gts_batch[i % local_batch_size]
            docs = docs_list[i % local_batch_size]
            doc_scores = self.ranker.run_query(new_query, docs)
            performance = self.evaluator.eval(gts,doc_scores,label=label)
            reward = self.reward_function(performance)

            reward_list.append(reward)
            for k,v in performance.items():
                metrics[k].append(v)

        return reward_list, metrics

    def knrm_env_step(self,query_exp,docs_list,gts_batch,label='PSCM'):
        '''
        :param query_exp: expanded_query_batch, unpadded
        :param docs: doc_batch, unpadded
        :param gts: ground truth scores
        :return:
        '''
        reward_list = []
        metrics = defaultdict(list)

        doc_size_list = []
        query_batch, doc_batch = [], []
        local_batch_size = len(docs_list)

        gts_expand_batch = []
        for i,new_query in enumerate(query_exp):
            q_index = i % local_batch_size
            docs = docs_list[q_index]
            gts = gts_batch[q_index]
            doc_size_list.append(len(docs))
            pad_new_query = pad_seq(new_query, self.max_expand_query_length)
            for gt,doc in zip(gts, docs):
                query_batch.append(pad_new_query)
                doc_batch.append(pad_seq(doc, self.max_doc_length))
                gts_expand_batch.append(gt)

        #max_expand_query_length = min(self.max_expand_query_length, )
        pad_query_batch = np.array(query_batch)
        pad_doc_batch = np.array(doc_batch)
        doc_scores = self.ranker(pad_query_batch, pad_doc_batch).view(-1)

        doc_scores = list(doc_scores.view(-1).detach().cpu().numpy())
        start = 0
        for i,dl in enumerate(doc_size_list):
            end = start + dl
            current_q_doc_scores = doc_scores[start:end]
            gts = gts_batch[i % local_batch_size]
            performance = self.evaluator.eval(gts, current_q_doc_scores, label=label)
            reward = self.reward_function(performance)
            reward_list.append(reward)
            for k, v in performance.items():
                metrics[k].append(v)
            start += dl
        return reward_list, metrics

    def bert_env_step(self,query_exp,docs_list,gts_batch,label='PSCM'):
        '''
        :param query_exp: expanded_query_batch, unpadded
        :param docs: doc_batch, unpadded
        :param gts: ground truth scores
        :return:
        '''
        reward_list = []
        metrics = defaultdict(list)

        doc_size_list = []
        query_batch, doc_batch = [], []
        local_batch_size = len(docs_list) #/ self.num_repeat


        for i,new_query in enumerate(query_exp):
            q_index = i % local_batch_size
            docs = docs_list[q_index]
            doc_size_list.append(len(docs))
            for doc in docs:
                query_batch.append(new_query)
                doc_batch.append(doc)

        pad_text_batch = [pad_seq_bert(bert_convert_ids(query,doc,self.bert_tokenizer), self.max_concat_text_length)\
                            for query,doc in zip(query_batch, doc_batch)]
        pad_text_batch = np.array(pad_text_batch)
        doc_scores = self.ranker(pad_text_batch)

        doc_scores = list(doc_scores.view(-1).detach().cpu().numpy())
        start = 0
        for i,dl in enumerate(doc_size_list):
            end = start + dl
            current_q_doc_scores = doc_scores[start:end]
            gts = gts_batch[i % local_batch_size]
            performance = self.evaluator.eval(gts, current_q_doc_scores, label=label)
            reward = self.reward_function(performance)
            reward_list.append(reward)
            for k, v in performance.items():
                metrics[k].append(v)
            start += dl

        return reward_list, metrics

    def reward_function(self,performance):
        reward = sum(map(lambda m:performance[m], self.rewards))
        return reward #+ performance['ndcg@3']#* 300

    def loss_function(self,cand_probs, reward_list, entropy):

        returns = Tensor2Varible(torch.FloatTensor(reward_list))
        if self.select_agent.training and self.mean_baseline and self.num_repeat > 1:
            mean_returns_pre = torch.mean(returns.view(-1, cand_probs.size(0)/self.num_repeat, 1), 0).view(-1, 1)#(n_repeat,bs,1)
            mean_returns = mean_returns_pre.repeat(self.num_repeat, 1)
            real_returns = nn.ReLU()(returns - mean_returns).detach()
        else:
            real_returns = returns

        log_pi = cand_probs.clamp(1e-8,1).log() #torch.log()
        sum_pi = torch.sum(log_pi,1).view(-1,1)


        rl_loss = - (sum_pi) * (real_returns)
        rl_loss = torch.sum(rl_loss)
        entropy_reg = torch.sum(entropy)
        total_loss = rl_loss - self.entropy_coef * entropy_reg#+ 0.1 * entropy#5 * entropy #+ ranker_loss #+ self.value_loss_alpha * value_loss #+ entropy
        pi = torch.sum(sum_pi)
        return total_loss, rl_loss, pi, returns, real_returns

    def predict(self,batch_samples, training_flag=True,label='PSCM'):
        self.select_agent.train(training_flag)
        self.select_agent.zero_grad()
        self.optimizer.zero_grad()

        query_ent_idx_batch = Tensor2Varible(torch.LongTensor(batch_samples['query_ent_idx_batch']))
        docs_batch = batch_samples['docs']
        cand_term_batch = Tensor2Varible(torch.LongTensor(batch_samples['query_cand_terms']))
        raw_query_batch = batch_samples['raw_query']
        gts_batch = batch_samples['gts']


        num_repeat = self.num_repeat if training_flag else 1
        cand_probs, select_probs, expand_idx, entropy = self.select_agent(query_ent_idx_batch, cand_term_batch,
                                                             num_repeat=num_repeat)  # TODO, remove repeat
        expand_idx = expand_idx.cpu().numpy()


        if self.ranker_name == 'BERT':
            new_query_batch = get_raw_text_bert(batch_samples['raw_query_text'], expand_idx, self.id2word)
        else:
            new_query_batch = get_raw_text(raw_query_batch, expand_idx, self.id2word)


        if self.ranker_name == 'BM25':
            reward, metrics = self.env_step(new_query_batch, docs_batch, gts_batch,label=label)

        elif self.ranker_name == 'KNRM':
            reward, metrics = self.knrm_env_step(new_query_batch, docs_batch, gts_batch, label=label)
        elif self.ranker_name == 'BERT':
            raw_doc_text = batch_samples['raw_doc_text']
            reward, metrics = self.bert_env_step(new_query_batch, raw_doc_text, gts_batch, label=label)


        if training_flag:

            total_loss, rl_loss, pi, reward, real_Reward = self.loss_function(select_probs, reward, entropy)
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(self.select_agent.parameters(), self.clip_grad)#self.clip_grad
            #torch.nn.utils.clip_grad_norm_(self.select_agent.parameters(), 0.1)
            self.optimizer.step()
            if self.use_scheduler:
                self.scheduler.step()
            reward = reward.detach().cpu().numpy()
            real_Reward = real_Reward.detach().cpu().numpy()
        else:
            total_loss = torch.FloatTensor([0.])
            rl_loss = torch.FloatTensor([0.])
            select_probs = torch.FloatTensor([0.])
            cand_probs = torch.FloatTensor([0.])
            pi = torch.sum(select_probs)
            log_pi = cand_probs.clamp(1e-8, 1).log()  # torch.log()
            entropy = - torch.sum(cand_probs * log_pi)
            real_Reward = reward

        #print('rewards:',reward)
        num_samples = len(raw_query_batch)
        entropy = torch.sum(entropy)

        return num_samples, total_loss,rl_loss, reward, real_Reward, metrics, pi, len(cand_term_batch[0]) * num_samples, entropy


    def run_epoch(self,data_loader,training_flag,print_process=False,epoch_num=0,label='PSCM'):
        """Run an epoch."""
        train_loss = 0
        samples_processed = 0
        total_reward,total_real_reward = 0,0
        total_metrics = defaultdict(list)
        total_pi = 0;total_entropy = 0
        total_rl_loss = 0
        print_avg_reward, print_avg_loss, print_samples = 0, 0, 0
        print_cand_size = 0;print_entropy = 0
        print_pi = 0
        s = datetime.datetime.now()
        iters_per_epoch = int(math.ceil(len(data_loader.dataset) * 1.0 / self.batch_size))

        for i,batch_samples in enumerate(data_loader):
            #torch.cuda.empty_cache()
            num_samples, total_loss,rl_loss, reward, real_Reward, metrics, pi, cand_size,entropy\
                = self.predict(batch_samples, training_flag=training_flag,label=label)


            # compute train loss
            samples_processed += num_samples
            train_loss += total_loss.item()
            total_pi += pi.item()
            total_reward += np.sum(reward)
            total_real_reward += np.sum(real_Reward)
            print_entropy += entropy.item()
            total_entropy += entropy.item()
            total_rl_loss += rl_loss.item()

            for k,metric_list in metrics.items():
                total_metrics[k].extend(metric_list)

            print_avg_reward += np.sum(reward)
            print_avg_loss += total_loss.item()

            print_samples += num_samples
            print_pi += pi.item()
            print_cand_size += cand_size

            if print_process and i % self.train_freq == 0:
                print_avg_reward /= print_samples
                print_avg_loss /= print_samples
                print_pi /= print_samples
                print_entropy /= print_samples
                print_cand_size /= print_samples
                e = datetime.datetime.now()
                g_step = (epoch_num - 1) * iters_per_epoch + i
                print(
                    "Step: [{}]\tEpoch: [{}/{}]\tSamples: [{}/{}]\tAvg Reward: {:.3f}\tTrain Loss: {:.5f}\tPrint_pi: {:.5f}".format(
                        g_step, epoch_num, self.num_epochs, samples_processed,
                        len(data_loader.dataset), print_avg_reward, print_avg_loss, print_pi))

                self.writer.add_scalar('Batch Reward', print_avg_reward, global_step=g_step)
                self.writer.add_scalar('Batch loss', print_avg_loss, global_step=g_step)
                self.writer.add_scalar('Batch Pi', print_pi, global_step=g_step)
                #self.writer.add_scalar('Batch Entropy', print_entropy, global_step=g_step)
                self.writer.add_scalar('Batch CandSize', print_cand_size, global_step=g_step)

                s = datetime.datetime.now()
                print_avg_reward, print_avg_loss, print_samples,print_pi,print_cand_size,print_entropy = 0, 0, 0, 0,0,0

        train_loss /= samples_processed
        total_pi /= samples_processed
        total_reward /= samples_processed
        total_real_reward /= samples_processed
        total_entropy /= samples_processed
        total_rl_loss /= samples_processed

        avg_metrics = {}
        for k, metric_list in total_metrics.items():
            avg_metrics[k] = np.mean(metric_list)
        return samples_processed, train_loss,total_rl_loss, total_reward, total_real_reward, avg_metrics, total_pi, total_entropy

    def fit(self):
        train_dataset = EntityDataset(self.config['train_addr'], self.common_resource)
        validation_dataset = EntityDataset(self.config['valid_addr'], self.common_resource)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate_fun)
        validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=self.collate_fun)

        train_loss = 0
        samples_processed = 0
        best_metric,last_metric = 0., 0,
        patience = self.patience
        train_ranker_epoch = 0
        # train loop
        print('==========================================================')
        print('=====================[Training START]=====================')
        for epoch in range(self.num_epochs):
            # train epoch
            self.nn_epoch = epoch
            s = datetime.datetime.now()
            sp, train_loss,rl_loss, avg_reward, avg_real_reward, avg_metrics, avg_pi,avg_entropy = self.run_epoch(train_loader,
                                                                                              training_flag=True,
                                                                                              print_process=True,
                                                                                              epoch_num=epoch + 1)
            samples_processed += sp
            e = datetime.datetime.now()

            avg_ndcg_10 = avg_metrics['ndcg@10']
            print("Total Epoch: [{}/{}]\tSamples: [{}/{}]\tAvg Reward: {:.3f}\tTrain Loss: {:.5f}\tTime: {}".format(
                epoch + 1, self.num_epochs, samples_processed,
                len(train_dataset) * self.num_epochs, avg_reward, train_loss, e - s))
            print("=======Training Metrics: NDCG@10: {:.3f}=======".format(avg_ndcg_10))
            self.writer.add_scalar('Epoch Real Reward', avg_real_reward, global_step=epoch)
            self.writer.add_scalar('Epoch Reward', avg_reward, global_step=epoch)
            self.writer.add_scalar('Epoch loss', train_loss, global_step=epoch)
            self.writer.add_scalar('Epoch RL loss', rl_loss, global_step=epoch)
            self.writer.add_scalar('Epoch Entropy', avg_entropy, global_step=epoch)

            print('\n=====================[Validation START]=====================')
            v_size, v_loss,v_rl_loss, v_avg_reward,v_real_avg_reward, v_avg_metrics,v_avg_pi,\
                    v_avg_entropy = self.run_epoch(validation_loader,training_flag=False)
            current_metric = v_avg_metrics['ndcg@10'] # or v_loss
            print('Message: ', self.message)
            print('Patience remains: ', patience)
            print_performance(v_avg_metrics,name="Validation metrics")

            self.writer.add_scalar('Valid Real Reward', v_real_avg_reward, global_step=epoch)
            self.writer.add_scalar('Valid Reward', v_avg_reward, global_step=epoch)
            self.writer.add_scalar('Valid Pi', v_avg_pi, global_step=epoch)
            self.writer.add_scalar('Valid NDCG@10', v_avg_metrics['ndcg@10'], global_step=epoch)
            self.writer.add_scalar('Valid NDCG@5', v_avg_metrics['ndcg@5'], global_step=epoch)
            self.writer.add_scalar('Valid NDCG@3', v_avg_metrics['ndcg@3'], global_step=epoch)
            self.writer.add_scalar('Valid NDCG@1', v_avg_metrics['ndcg@1'], global_step=epoch)
            self.writer.add_scalar('Valid MAP', v_avg_metrics['map'], global_step=epoch)
            self.writer.add_scalar('Valid Entropy', v_avg_entropy, global_step=epoch)
            # Early stopping
            if current_metric > best_metric:
                print('Got better result, save to %s' % self.saveModeladdr)
                best_metric = current_metric
                patience = self.patience
                self.save(message=self.message)
            elif current_metric <= last_metric * self.cost_threshold:
                patience -= 1
            last_metric = current_metric
            if patience < 0:
                print('patience runs out...')
                break

            
            if self.train_ranker_bool and epoch > 0 and epoch % self.train_ranker_freq == 0:
                if self.ranker_name == 'KNRM':
                    total_ranker_loss = self.train_knrm(train_dataset.qid2cand_terms)
                elif self.ranker_name == 'BERT':
                    total_ranker_loss = self.train_bert(train_dataset.qid2cand_terms)
                self.writer.add_scalar('Ranker loss', total_ranker_loss, global_step=train_ranker_epoch)
                print('Finish training: ', total_ranker_loss)
                train_ranker_epoch += 1
            torch.cuda.empty_cache()

        print('Patience___: ', patience)
        print("All done, exiting...")

    def test(self,ground_truth='HUMAN',data='test'):
        '''
        :param ground_truth:
        :param data: test or valid
        :return:
        '''
        self.evaluator.rel_threshold = 1
        test_dataset = EntityDataset(self.config[data+'_addr'], common_resource=self.common_resource,ground_truth=ground_truth)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,collate_fn=self.collate_fun)
        start_time = time.time()
        test_size, loss,rl_loss, avg_reward,real_avg_reward, avg_metrics,avg_pi,avg_entropy = self.run_epoch(test_loader, training_flag=False, label=ground_truth)
        end_time = time.time()
        print('Cost time: ', (end_time - start_time))
        print('Test size: %.3f' % test_size)
        print('Avg_reward: %.3f' % avg_reward)
        print_performance(avg_metrics, name="Testing metrics: " + ground_truth)

    def save(self, message):
        """
        Save model.
        Args
            models_dir: path to directory for saving NN models.
        """
        if not os.path.isdir(self.saveModeladdr):
            os.makedirs(self.saveModeladdr)

        filename = "{}_epoch_{}".format(message, self.nn_epoch) + '.pth'
        fileloc = os.path.join(self.saveModeladdr, filename)
        with open(fileloc, 'wb') as file:
            # excluding the ranker
            save_dict = {'state_dict': self.select_agent.state_dict(),'config':self.config}
            if self.train_ranker_bool and self.ranker_name != 'BM25':
                save_dict['ranker_dict'] = self.ranker.state_dict()
            torch.save(save_dict, file)
            #,'dcue_dict': self.__dict__

    def load(self, model_addr):
        """
        Load a previously trained model.
        Args
            model_dir: directory where models are saved.
            epoch: epoch of model to load.
        """
        print('Resuming model from: ', model_addr)
        with open(model_addr, 'rb') as model_dict:
            checkpoint = torch.load(model_dict,map_location=self.loc)

        self.select_agent.load_state_dict(checkpoint['state_dict'])
        if self.train_ranker_bool and self.ranker_name != 'BM25':
            self.ranker.load_state_dict(checkpoint['ranker_dict'])

    def train_knrm(self,qid2cand_terms):
        freeze_ranker(self.ranker, requires_grad=True)
        self.ranker.train()
        self.select_agent.eval()
        train_loader = RankerDataGenerator(self.config,self.word2id,qid2cand_terms,self.use_knowledge,self.is_descrption)

        total_loss = 0.
        print('=============Training KNRM=============')
        for i, (query_batch,query_ent_batch, doc_pos_batch, doc_neg_batch, candidate_batch) in enumerate(train_loader.ranking_pair_reader(self.config['ranker_bs'])):
            self.ranker.zero_grad()
            self.ranker_optimizer.zero_grad()
            query_batch_tensor = Tensor2Varible(torch.LongTensor(query_ent_batch))
            cand_term_batch_tensor = Tensor2Varible(torch.LongTensor(candidate_batch))
            cand_probs, expand_idx, avg_reward = self.select_agent(query_batch_tensor, cand_term_batch_tensor)
            expand_idx = expand_idx.cpu().numpy()

            new_query_batch = get_raw_text(query_batch, expand_idx, self.id2word)
            max_q_len = np.max(map(lambda d:len(d), new_query_batch))
            pad_new_query = np.array([pad_seq(new_query, max_q_len) for new_query in new_query_batch])

            pos_score = self.ranker(pad_new_query, doc_pos_batch)
            neg_score = self.ranker(pad_new_query, doc_neg_batch)

            rank_loss = torch.sum(torch.clamp(1.0 - pos_score + neg_score, min=0))

            rank_loss.backward()
            self.ranker_optimizer.step()
            total_loss += rank_loss.item()
            if i % 100 == 0:
                print('Batch id:%d\t Batch loss: %.3f\tTotal loss: %.3f' % (i, rank_loss.item(), total_loss/(i+1)))
                #avg 1090, bs=200, i = 10


        print('=============Finish Training KNRM=============')
        return total_loss/(i+1)

    def train_bert(self,qid2cand_terms):
        freeze_ranker(self.ranker, requires_grad=True)
        self.ranker.train()
        self.select_agent.eval()
        train_loader = BERTDataGenerator(self.config,self.word2id,qid2cand_terms,self.use_knowledge,self.is_descrption)

        total_loss = 0.
        print('=============Training BERT=============')
        for i, (query_batch,raw_query_text,query_ent_batch,candidate_batch, pos_doc_batch, neg_doc_batch) in enumerate(train_loader.bert_pair_reader(self.config['ranker_bs'])):
            self.ranker.zero_grad()
            self.ranker_optimizer.zero_grad()
            query_batch_tensor = Tensor2Varible(torch.LongTensor(query_ent_batch))
            cand_term_batch_tensor = Tensor2Varible(torch.LongTensor(candidate_batch))
            cand_probs, expand_idx, avg_reward = self.select_agent(query_batch_tensor, cand_term_batch_tensor)
            expand_idx = expand_idx.cpu().numpy()
            #print('query_batch:',len(query_batch),expand_idx.shape)

            new_query_batch = get_raw_text_bert(raw_query_text, expand_idx, self.id2word)

            pad_pos_text_batch = np.array([
                pad_seq_bert(bert_convert_ids(query, doc, self.bert_tokenizer), self.max_concat_text_length) \
                for query, doc in zip(new_query_batch, pos_doc_batch)])
            pos_scores = self.ranker(pad_pos_text_batch)

            pad_neg_text_batch = np.array([
                pad_seq_bert(bert_convert_ids(query, doc, self.bert_tokenizer), self.max_concat_text_length) \
                for query, doc in zip(new_query_batch, neg_doc_batch)])
            neg_scores = self.ranker(pad_neg_text_batch)

            rank_loss = torch.sum(torch.clamp(1.0 - pos_scores + neg_scores, min=0))

            rank_loss.backward()
            self.ranker_optimizer.step()
            total_loss += rank_loss.item()
            if i % 1 == 0:
                print('Batch id:%d\t Batch loss: %.3f\tTotal loss: %.3f' % (i, rank_loss.item(), total_loss/(i+1)))

        print('=============Finish Training BERT=============')
        freeze_ranker(self.ranker, requires_grad=False)
        return total_loss/(i+1)














            






