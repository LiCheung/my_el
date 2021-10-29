# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_loader.py

@time: 2019/5/8 22:07

@desc:

"""
import os
import copy
import numpy as np
import codecs
import jieba
from random import sample
from keras.utils import Sequence, to_categorical
from keras_bert import Tokenizer
from tqdm import tqdm

from utils.other import pad_sequences_1d, pad_sequences_2d
from utils.io import pickle_load, format_filename
from config import PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME, DEV_DATA_FILENAME, TEST_DATA_FILENAME, \
    MENTION_TO_ENTITY_FILENAME, TEST_FINAL_DATA_FILENAME


def load_data(data_type):
    if data_type == 'train':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME))
    elif data_type == 'dev':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_FILENAME))
    elif data_type == 'test':
        data = pickle_load(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME))
    else:
        raise ValueError('data tye not understood: {}'.format(data_type))
    return data


class LinkDataGenerator(Sequence):
    """
    Data Generator for entity linking model
    """

    def __init__(self, data_type, word_vocab, mention_to_entity, entity_desc, entity_id, batch_size, max_desc_len,
                 max_erl_len, use_relative_pos=False, n_neg=1, omit_one_cand=True, shuffle=True):
        self.data_type = data_type
        self.data = load_data(data_type)
        self.data_size = len(self.data)
        self.batch_size = batch_size
        self.indices = np.arange(self.data_size)
        self.steps = int(np.ceil(self.data_size / self.batch_size))

        self.word_vocab = word_vocab
        self.mention_to_entity = mention_to_entity
        self.entity_desc = entity_desc
        self.entity_id = entity_id
        self.max_desc_len = max_desc_len

        self.use_relative_pos = use_relative_pos  # use relative position (to mention) as model's input
        self.max_erl_len = max_erl_len
        self.n_neg = n_neg      # how many negative sample
        self.omit_one_cand = omit_one_cand  # exclude those samples whose mention only has one candidate entity
        self.shuffle = shuffle

    def __len__(self):
        return self.steps

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_word_ids(self, word_cut):
        word_ids = []
        for word in word_cut:
            if word==' ' or self.word_vocab.get(word)==None:
                continue
            word_ids.append(self.word_vocab.get(word))  # all char in one word share the same word embedding
        return word_ids

    def __getitem__(self, index):
        batch_index = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        batch_text_ids, batch_begin, batch_end, batch_rel_pos = [], [], [], []
        batch_pos_desc_ids, batch_neg_desc_ids = [], []
        batch_labels = [], []

        for i in batch_index:
            data = self.data[i]
            raw_text = data['text']
            # print(data)
            word_cut = jieba.lcut(raw_text)
            text_ids = self.get_word_ids(word_cut)
            # print('text_ids: ', end="")
            # print(text_ids)

            if 'mention_data' in data:
                for mention in data['mention_data']:
                    begin = mention[1]
                    end = begin
                    if self.use_relative_pos:
                        relative_pos = self.get_relative_pos(begin, end, len(text_ids))

                    pos_ent = mention[2]    # positive entity
                    cand_ents = copy.deepcopy(self.mention_to_entity[mention[0]])
                    while pos_ent in cand_ents:
                        cand_ents.remove(pos_ent)

                    # generate negative samples
                    if len(cand_ents) == 0:
                        if self.omit_one_cand:
                            continue
                        else:
                            neg_ents = sample(self.entity_desc.keys(), self.n_neg)
                    elif len(cand_ents) < self.n_neg:
                        neg_ents = cand_ents + sample(self.entity_desc.keys(), self.n_neg - len(cand_ents))
                    else:
                        neg_ents = sample(cand_ents, self.n_neg)

                    # pos_ent为屏幕名,desc索引为id,id索引为screenname
                    pos_desc_cut = jieba.lcut(self.entity_desc[pos_ent])
                    pos_desc_ids = self.get_word_ids(pos_desc_cut)
                    for neg_ent in neg_ents:
                        if type(neg_ent) == '<class \'str\'>':
                            neg_ent = self.entity_id[neg_ent]
                        neg_desc_cut = jieba.lcut(self.entity_desc[neg_ent])
                        neg_desc_ids = self.get_word_ids(neg_desc_cut)
                        batch_neg_desc_ids.append(neg_desc_ids)
                        batch_pos_desc_ids.append(pos_desc_ids)

                    for _ in range(self.n_neg):
                        batch_text_ids.append(text_ids)
                        batch_begin.append([begin])
                        batch_end.append([end])
                        if self.use_relative_pos:
                            batch_rel_pos.append(relative_pos)
        batch_inputs = []
        batch_text_ids = pad_sequences_1d(batch_text_ids)
        batch_inputs.append(batch_text_ids)
        batch_begin = np.array(batch_begin)
        batch_end = np.array(batch_end)
        batch_inputs.extend([batch_begin, batch_end])

        if self.use_relative_pos:
            batch_rel_pos = pad_sequences_1d(batch_rel_pos)
            batch_inputs.append(batch_rel_pos)
        batch_pos_desc_ids = pad_sequences_1d(batch_pos_desc_ids, max_len=self.max_desc_len)
        batch_neg_desc_ids = pad_sequences_1d(batch_neg_desc_ids, max_len=self.max_desc_len)
        batch_inputs.extend([batch_pos_desc_ids, batch_neg_desc_ids])
        return batch_inputs, None

    def get_relative_pos(self, begin, end, text_len):
        relative_pos = []
        for i in range(text_len):
            if i < begin:
                relative_pos.append(max(i - begin + self.max_erl_len, 1))   # transform to id
            elif begin <= i <= end:
                relative_pos.append(0 + self.max_erl_len)
            else:
                relative_pos.append(min(i - end+self.max_erl_len, 2*self.max_erl_len-1))
        return relative_pos

