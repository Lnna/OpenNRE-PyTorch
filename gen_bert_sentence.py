import numpy as np
import os
import json
import time

in_path = "./mnre_data/176rels_data/new_data/"
out_path = "./mnre_data/176rels_data/need_data/"
case_sensitive = False
if not os.path.exists(out_path):
    os.mkdir(out_path)
train_file_name = in_path + 'train.json'
test_file_name = in_path + 'test.json'
word_file_name = in_path + 'word_vec.json'
rel_file_name = in_path + 'rel2id.json'

import logging
from stanfordcorenlp.corenlp import StanfordCoreNLP


class StanfordNlp(StanfordCoreNLP):
    def __init__(self, path_or_host, port=None, memory='4g', lang='en', timeout=1500, quiet=True,
                 logging_level=logging.WARNING):
        super(StanfordNlp, self).__init__(path_or_host, lang=lang)

    def pos_tag(self, sentence):
        r_dict = self._request('pos', sentence)
        words = []
        tags = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                words.append(token['word'])
                tags.append(token['pos'])
        return list(zip(words, tags))


def find_pos(sentence, head, tail):
    def find(sentence, entity):
        p = sentence.find(' ' + entity + ' ')
        if p == -1:
            if sentence[:len(entity) + 1] == entity + ' ':
                p = 0
            elif sentence[-len(entity) - 1:] == ' ' + entity:
                p = len(sentence) - len(entity)
            else:
                p = 0
        else:
            p += 1
        return p

    sentence = ' '.join(sentence.split())
    p1 = find(sentence, head)
    p2 = find(sentence, tail)
    words = sentence.split()
    cur_pos = 0
    pos1 = -1
    pos2 = -1
    for i, word in enumerate(words):
        if cur_pos == p1:
            pos1 = i
        if cur_pos == p2:
            pos2 = i
        cur_pos += len(word) + 1
    return pos1, pos2


def init(file_name, word_vec_file_name, rel2id_file_name, max_length=120, case_sensitive=False, is_training=True):
    if file_name is None or not os.path.isfile(file_name):
        raise Exception("[ERROR] Data file doesn't exist")
    if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
        raise Exception("[ERROR] Word vector file doesn't exist")
    if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
        raise Exception("[ERROR] rel2id file doesn't exist")

    print("Loading data file...")
    ori_data = json.load(open(file_name, "r"))
    print("Finish loading")

    if not case_sensitive:
        print("Eliminating case sensitive problem...")
        for i in ori_data:
            i['sentence'] = i['sentence'].lower()
            i['head']['word'] = i['head']['word'].lower()
            i['tail']['word'] = i['tail']['word'].lower()
        print("Finish eliminating")

    # sorting
    print("Sorting data...")
    ori_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
    print("Finish sorting")

    sen_tot = len(ori_data)
    print('sentence totally:{}'.format(sen_tot))
    sentences=[]
    for i in range(len(ori_data)):
        if i % 1000 == 0:
            print(i)
        sen = ''.join(ori_data[i]['sentence'].split())
        # if len(sen)<max_length:
        #     sen=sen+' '+' '.join(['[PAD]']*(max_length-len(sen)))
        # else:
        #     sen=sen[:max_length]
        sentences.append(sen)
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "test"
    np.save(os.path.join(out_path, name_prefix + '_sentences.npy'), sentences)

init(train_file_name, word_file_name, rel_file_name, max_length=200, case_sensitive=False, is_training=True)
init(test_file_name, word_file_name, rel_file_name, max_length=200, case_sensitive=False, is_training=False)
