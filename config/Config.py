#coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import pickle as pc
from pytorch_pretrained_bert import BertTokenizer, BertModel

# sys.path.append('/home/nana/Documents/pycharmforlinux/mParser')
# sys.path.append('/home/lnn/Documents/private/mParser')
# from src.gen_mediate_para import hs_parse
# print(hs_parse([('NN','人们'),('VV','爱'),('NN','小明')]))

device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
# device="cpu"
# torch.cuda.set_device(0)
def to_var(x):
    return Variable(torch.from_numpy(x).to(device))

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0 

class Config(object):
    def __init__(self):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = './mnre_data/176rels_data/need_data/'
        # self.data_path = '/home/lnn/Documents/OpenNRE-Ina/OpenNRE-PyTorch/data'
        self.use_bag = True
        self.use_gpu = True
        self.is_training = True
        self.max_length = 120
        self.pos_num = 2 * self.max_length
        #NYT
        # self.num_classes = 53
        # mnre
        self.num_classes = 176
        self.hidden_size = 230
        self.pos_size = 5
        self.max_epoch = 15
        self.opt_method = 'SGD'
        self.optimizer = None
        self.learning_rate = 0.1
        self.weight_decay = 1e-5
        self.drop_prob = 0.5
        self.checkpoint_dir = './mnre_data/176rels_data/merge_checkpoint'
        self.test_result_dir = './mnre_data/176rels_data/merge_test_result'
        self.save_epoch = 1
        self.test_epoch = 1
        self.pretrain_model = None
        self.trainModel = None
        self.testModel = None
        self.batch_size = 30
        self.word_size = 50
        self.window_size = 3
        self.epoch_range = None
        self.tokenizer = BertTokenizer.from_pretrained('/media/sda1/nana/bert-rel/chinese_L-12_H-768_A-12')
        self.bert_model = BertModel.from_pretrained('/media/sda1/nana/bert-rel/')
        self.bert_model.eval()
        self.bert_model.to(device)


    def set_data_path(self, data_path):
        self.data_path = data_path
    def set_max_length(self, max_length):
        self.max_length = max_length
        self.pos_num = 2 * self.max_length
    def set_num_classes(self, num_classes):
        self.num_classes = num_classes
    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
    def set_window_size(self, window_size):
        self.window_size = window_size
    def set_pos_size(self, pos_size):
        self.pos_size = pos_size
    def set_word_size(self, word_size):
        self.word_size = word_size
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_opt_method(self, opt_method):
        self.opt_method = opt_method
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch
    def set_save_epoch(self, save_epoch):
        self.save_epoch = save_epoch
    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model
    def set_is_training(self, is_training):
        self.is_training = is_training
    def set_use_bag(self, use_bag):
        self.use_bag = use_bag
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range


    def load_train_data(self):
        print("Reading training data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_train_word = np.load(os.path.join(self.data_path, 'train_word.npy'))
        self.data_train_pos1 = np.load(os.path.join(self.data_path, 'train_pos1.npy'))
        self.data_train_pos2 = np.load(os.path.join(self.data_path, 'train_pos2.npy'))
        self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))
        if self.use_bag:
            self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
            self.data_train_label = np.load(os.path.join(self.data_path, 'train_bag_label.npy'))
            self.data_train_scope = np.load(os.path.join(self.data_path, 'train_bag_scope.npy'))
        else:
            self.data_train_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
            self.data_train_scope = np.load(os.path.join(self.data_path, 'train_ins_scope.npy'))

        self.train_sentences=np.load(os.path.join(self.data_path,'train_sentences.npy'))

        print("Finish reading")
        self.train_order = list(range(len(self.data_train_label)))
        self.train_batches = len(self.data_train_label) // self.batch_size
        if len(self.data_train_label) % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_test_word = np.load(os.path.join(self.data_path, 'test_word.npy'))
        self.data_test_pos1 = np.load(os.path.join(self.data_path, 'test_pos1.npy'))
        self.data_test_pos2 = np.load(os.path.join(self.data_path, 'test_pos2.npy'))
        self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
        if self.use_bag:
            self.data_test_label = np.load(os.path.join(self.data_path, 'test_bag_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.data_path, 'test_bag_scope.npy'))
        else:
            self.data_test_label = np.load(os.path.join(self.data_path, 'test_ins_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.data_path, 'test_ins_scope.npy'))

        self.test_sentences=np.load(os.path.join(self.data_path,'test_sentences.npy'))

        print("Finish reading")
        self.test_batches = len(self.data_test_label) // self.batch_size
        if len(self.data_test_label) % self.batch_size != 0:
            self.test_batches += 1

        self.total_recall = self.data_test_label[:, 1:].sum()

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config = self)
        if self.pretrain_model != None:
            self.trainModel.load_state_dict(torch.load(self.pretrain_model))
        self.trainModel.to(device)
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr = self.learning_rate, lr_decay = self.lr_decay, weight_decay = self.weight_decay)
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        print("Finish initializing")
              
    def set_test_model(self, model):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config = self)
        self.testModel.to(device)
        self.testModel.eval()
        print("Finish initializing")

    def get_train_batch(self, batch):
        input_scope = np.take(self.data_train_scope, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_train_word[index, :]
        self.batch_pos1 = self.data_train_pos1[index, :]
        self.batch_pos2 = self.data_train_pos2[index, :]
        self.batch_mask = self.data_train_mask[index, :]    
        self.batch_label = np.take(self.data_train_label, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
        # print('batch label shape {}'.format(self.batch_label.shape))
        self.batch_attention_query = self.data_query_label[index]
        self.batch_scope = scope

        self.batch_lstm_hs = []
        lstm_mod = 1000
        lstm_dir = '/media/sda1/nana/opennre-pytorch/mnre_data/176rels_data/need_data/f189_lstm_parse'
        tmp = dict()
        for i in index:
            tlist = tmp.get(i // lstm_mod + 1, [])
            tlist.append(i)
            tmp[i // lstm_mod + 1] = tlist

        batch_dict = dict()
        for k, v in tmp.items():
            lstm_dic = pc.load(open(os.path.join(lstm_dir, 'train_{}.pc'.format(k)), mode='rb'))
            for i in v:
                batch_dict[i] = lstm_dic[i]
        for i in index:
            self.batch_lstm_hs.append(batch_dict[i])
        self.batch_lstm_hs = Variable(torch.from_numpy(np.array(self.batch_lstm_hs)).float().to(device))

        batch_sentences=self.train_sentences[index]
        self.batch_bert =[]
        for sen in batch_sentences:
            tokenized_text = self.tokenizer.tokenize(sen)
            if len(tokenized_text)<self.max_length:
                tokenized_text.extend(['[PAD]']*(self.max_length-len(tokenized_text)))
            else:
                tokenized_text=tokenized_text[:self.max_length]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            segments_ids = [0]*self.max_length

            # 将 inputs 转为 PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            # print(tokens_tensor.size())
            segments_tensors = torch.tensor([segments_ids])
            # print(segments_tensors.size())
            tokens_tensor = tokens_tensor.to(device)
            segments_tensors = segments_tensors.to(device)

            # 得到每一层的 hidden states
            with torch.no_grad():
                encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
            self.batch_bert.append(encoded_layers.squeeze(0)[-1].unsqueeze(0))
        self.batch_bert=Variable(torch.cat(self.batch_bert,dim=0))
    def get_test_batch(self, batch):
        input_scope = self.data_test_scope[batch * self.batch_size : (batch + 1) * self.batch_size]
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_test_word[index, :]
        self.batch_pos1 = self.data_test_pos1[index, :]
        self.batch_pos2 = self.data_test_pos2[index, :]
        self.batch_mask = self.data_test_mask[index, :]
        self.batch_scope = scope

        self.batch_lstm_hs = []
        lstm_mod = 1000
        lstm_dir = '/media/sda1/nana/opennre-pytorch/mnre_data/176rels_data/need_data/f189_lstm_parse'
        tmp = dict()
        for i in index:
            tlist = tmp.get(i // lstm_mod + 1, [])
            tlist.append(i)
            tmp[i // lstm_mod + 1] = tlist

        batch_dict = dict()
        for k, v in tmp.items():
            lstm_dic = pc.load(open(os.path.join(lstm_dir, 'test_{}.pc'.format(k)), mode='rb'))
            for i in v:
                batch_dict[i] = lstm_dic[i]
        for i in index:
            self.batch_lstm_hs.append(batch_dict[i])

        self.batch_lstm_hs = Variable(torch.from_numpy(np.array(self.batch_lstm_hs)).float().to(device))

        batch_sentences = self.test_sentences[index]
        self.batch_bert = []
        # print("batch_sentences")
        # print(len(batch_sentences))
        for sen in batch_sentences:
            tokenized_text = self.tokenizer.tokenize(sen)
            if len(tokenized_text) < self.max_length:
                tokenized_text.extend(['[PAD]'] * (self.max_length - len(tokenized_text)))
            else:
                tokenized_text = tokenized_text[:self.max_length]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            segments_ids = [0] * self.max_length

            # 将 inputs 转为 PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            # print(tokens_tensor.size())
            segments_tensors = torch.tensor([segments_ids])
            # print(segments_tensors.size())
            tokens_tensor = tokens_tensor.to(device)
            segments_tensors = segments_tensors.to(device)

            # 得到每一层的 hidden states
            with torch.no_grad():
                encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)
            self.batch_bert.append(encoded_layers.squeeze(0)[-1].unsqueeze(0))
        self.batch_bert = Variable(torch.cat(self.batch_bert, dim=0))

    def train_one_step(self):
        self.trainModel.embedding.word = to_var(self.batch_word)
        self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
        self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
        self.trainModel.encoder.mask = to_var(self.batch_mask)
        self.trainModel.selector.scope = self.batch_scope
        self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
        # print('attention_query shape {}'.format(self.trainModel.selector.attention_query.shape))
        self.trainModel.selector.label = to_var(self.batch_label)
        self.trainModel.classifier.label = to_var(self.batch_label)
        # print(self.trainModel.classifier.label)
        self.optimizer.zero_grad()
        loss, _output = self.trainModel()
        loss.backward()
        self.optimizer.step()
        for i, prediction in enumerate(_output):
            if self.batch_label[i] == 0:
                self.acc_NA.add(prediction.cpu().numpy() == self.batch_label[i])
            else:
                self.acc_not_NA.add(prediction.cpu().numpy() == self.batch_label[i])
            self.acc_total.add(prediction.cpu().numpy() == self.batch_label[i])
        # return loss.data[0]
        return loss.item()

    def test_one_step(self):
        self.testModel.embedding.word = to_var(self.batch_word)
        self.testModel.embedding.pos1 = to_var(self.batch_pos1)
        self.testModel.embedding.pos2 = to_var(self.batch_pos2)
        self.testModel.encoder.mask = to_var(self.batch_mask)
        self.testModel.selector.scope = self.batch_scope
        return self.testModel.test()

    def train(self):
        self.cur_epoch = 0
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        else:
            self.cur_epoch=1
            model_path=os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(self.cur_epoch-1))
            self.trainModel.load_state_dict(torch.load(model_path))
        best_auc = 0.0
        best_p = None
        best_r = None
        best_epoch = 0
        for epoch in range(self.cur_epoch,self.max_epoch):
            print('Epoch ' + str(epoch) + ' starts...')
            if epoch%3==0 and epoch!=0:
                self.set_learning_rate(self.learning_rate/2.6)
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            np.random.shuffle(self.train_order)
            for batch in range(self.train_batches):
                # print('total batches:{} now batch:{}'.format(self.train_batches,batch))
                self.get_train_batch(batch)
                loss = self.train_one_step()
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))    
                sys.stdout.flush()
            if (epoch + 1) % self.save_epoch == 0:
                print('Epoch ' + str(epoch) + ' has finished')
                print('Saving model...')
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
                torch.save(self.trainModel.state_dict(), path)
                print('Have saved model to ' + path)
            if (epoch + 1) % self.test_epoch == 0:
                self.testModel = self.trainModel
                auc, pr_x, pr_y = self.test_one_epoch()
                if auc > best_auc:
                    best_auc = auc
                    best_p = pr_x
                    best_r = pr_y
                    best_epoch = epoch
        print("Finish training")
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")
    def test_one_epoch(self):
        test_score = []
        for batch in tqdm(range(self.test_batches)):
            self.get_test_batch(batch)
            batch_score = self.test_one_step()
            test_score = test_score + batch_score
        test_result = []
        for i in range(len(test_score)):
            for j in range(1, len(test_score[i])):
                test_result.append([self.data_test_label[i][j], test_score[i][j]])
        test_result = sorted(test_result, key = lambda x: x[1])
        test_result = test_result[::-1]
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / self.total_recall)
            # if pr_x[-1] > 0.60:
        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        print("auc: ", auc)
        return auc, pr_x, pr_y
    def test(self):
        best_epoch = None
        best_auc = 0.0
        best_p = None
        best_r = None
        for epoch in self.epoch_range:
            path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
            if not os.path.exists(path):
                continue
            print("Start testing epoch %d" % (epoch))
            self.testModel.load_state_dict(torch.load(path))
            auc, p, r = self.test_one_epoch()
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_p = p
                best_r = r
            print("Finish testing epoch %d" % (epoch))
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")

    def predict(self,epoch,store_path='ori_predict'):
        path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
        if not os.path.exists(path):
            return
        self.testModel.load_state_dict(torch.load(path))
        test_score = []
        for batch in tqdm(range(self.test_batches)):
            self.get_test_batch(batch)
            batch_score = self.test_one_step()
            test_score = test_score + batch_score
        np.save(os.path.join(self.test_result_dir, store_path+'_res.npy'),test_score)
        if not os.path.exists(os.path.join(self.test_result_dir, 'true_label_res.npy')):
            np.save(os.path.join(self.test_result_dir, 'true_label_res.npy'),self.data_test_label)
        if not os.path.exists(os.path.join(self.test_result_dir, 'data_test_scope.npy')):
            np.save(os.path.join(self.test_result_dir, 'data_test_scope.npy'),self.data_test_scope)

