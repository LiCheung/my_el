# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: preprocess.py

@time: 2019/5/8 20:37

@desc:

"""

import os
import json
from collections import defaultdict
from tqdm import tqdm
import jieba
import numpy as np
from config import RAW_DATA_DIR,PROCESSED_DATA_DIR, LOG_DIR, SUBMIT_DIR, MODEL_SAVED_DIR, KB_FILENAME, MENTION_TO_ENTITY_FILENAME, \
    ENTITY_TO_MENTION_FILENAME, ENTITY_DESC_FILENAME, ENTITY_ID_FILENAME, ENTITY_TYPE_FILENAME, CCKS_TRAIN_FILENAME, VOCABULARY_TEMPLATE, \
    IDX2TOKEN_TEMPLATE, TRAIN_DATA_FILENAME, DEV_DATA_FILENAME, TEST_DATA_FILENAME, CCKS_TEST_FILENAME, \
    EMBEDDING_MATRIX_TEMPLATE, CCKS_TEST_FINAL_FILENAME, TEST_FINAL_DATA_FILENAME, IMG_DIR, M2E_FILENAME
from utils.io import pickle_dump, format_filename
from utils.embedding import train_w2v


def load_kb_data(kb_file):
    """process knowledge base file"""
    entity_desc = defaultdict()
    entity_id = defaultdict()
    with open(kb_file, encoding="utf-8") as reader:
        kb_data_all = json.load(reader)
        print(len(kb_data_all))
        for kb_data in kb_data_all:
            desc = kb_data['userDescription'].lower()
            # if not desc:
            #     continue
            entity_desc[str(kb_data['userId'])] = desc
            entity_id[kb_data['userScreenName']] = str(kb_data['userId'])
        return entity_desc,entity_id

def load_m2e(m2e_file):
    """process mention to entity file"""
    mention_to_entity = defaultdict(list)
    with open(m2e_file, encoding="utf-8") as reader:
        data = json.load(reader)
        mention = data[0]['lastName']
        for line in data:
            if mention == line['lastName']:
                if line['userId'] not in mention_to_entity[mention]:
                    mention_to_entity[mention].append(str(line['userId']))
            else:
                mention = line['lastName']
                mention_to_entity[mention].append(str(line['userId']))
        return mention_to_entity

def load_train_data(erl_file):
    train_data = []
    with open(erl_file, encoding="utf-8") as reader:
        for line in tqdm(reader):
            data = line.split('\t')
            erl_text = data[5].lower()
            mention_data = []
            mention_text, offset, entity = data[3].lower(), int(data[1]), data[4]
            if erl_text[offset: offset + len(mention_text)] != mention_text:
                offset = erl_text.find(mention_text)
            erl_text_cut = jieba.lcut(erl_text)
            i = 0
            j = 0
            for word in erl_text_cut:
                if word == ' ':
                    j+=1
                if i == offset:
                    offset = erl_text_cut.index(word) - j
                    break
                for w in word:
                    i += 1

            mention_data.append((mention_text, offset, entity_id[entity]))
            train_data.append({'text': erl_text, 'mention_data': mention_data})
    return train_data

def load_test_data(erl_file):
    test_data = []
    with open(erl_file, encoding="utf-8") as reader:
        for line in tqdm(reader):
            data = line.split('\t')

            erl_text = data[5].lower()
            mention_text, offset, entity = data[3].lower(), int(data[1]), data[4]
            if erl_text[offset: offset + len(mention_text)] != mention_text:
                offset = erl_text.find(mention_text)
            erl_text_cut = jieba.lcut(erl_text)
            i = 0
            j = 0
            for word in erl_text_cut:
                if word == ' ':
                    j += 1
                if i == offset:
                    offset = erl_text_cut.index(word) - j
                    break
                for w in word:
                    i += 1

            test_data.append({'text_id': data[0], 'text': data[5].lower(), 'raw_text': data[5], 'entity':data[4], 'mention': data[3], 'begin':offset})
    return test_data

def load_bichar_vocab_and_corpus(entity_desc, train_data, min_count=2):
    bichars = dict()
    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        bigrams = []
        for i in range(len(desc)):
            c = desc[i] + '</end>' if i == len(desc) - 1 else desc[i:i + 2]
            bigrams.append(c)
            bichars[c] = bichars.get(c, 0) + 1
        corpus.append(bigrams)
    for data in tqdm(iter(train_data)):
        bigrams = []
        for i in range(len(data['text'])):
            c = data['text'][i] + '</end>' if i == len(data['text']) - 1 else data['text'][i:i + 2]
            bigrams.append(c)
            bichars[c] = bichars.get(c, 0) + 1
        corpus.append(bigrams)
    bichars = {i: j for i, j in bichars.items() if j >= min_count}
    idx2bichar = {i + 2: j for i, j in enumerate(bichars)}  # 0: mask, 1: padding
    bichar2idx = {j: i for i, j in idx2bichar.items()}
    return bichar2idx, idx2bichar, corpus


def load_word_vocab_and_corpus(entity_desc, train_data, min_count=2):
    words = dict()
    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        desc_cut = jieba.lcut(desc)
        for w in desc_cut:
            words[w] = words.get(w, 0) + 1
        corpus.append(desc_cut)
    for data in tqdm(iter(train_data)):
        text_cut = jieba.lcut(data['text'])
        for w in text_cut:
            words[w] = words.get(w, 0) + 1
        corpus.append(text_cut)
    words = {i: j for i, j in words.items() if j >= min_count}
    idx2word = {i + 2: j for i, j in enumerate(words)}  # 0: mask, 1: padding
    word2idx = {j: i for i, j in idx2word.items()}
    return word2idx, idx2word, corpus


def load_charpos_vocab_and_corpus(char2idx, entity_desc, train_data):
    """build position aware character vocabulary by assign 4 positional tags: <B> <M> <E> <S>"""
    charpos2idx = {'<B>': 2, '<M>': 3, '<E>': 4, '<S>': 5}
    for c in char2idx.keys():
        charpos2idx[c+'<B>'] = len(charpos2idx) + 2
        charpos2idx[c+'<M>'] = len(charpos2idx) + 2
        charpos2idx[c+'<E>'] = len(charpos2idx) + 2
        charpos2idx[c+'<S>'] = len(charpos2idx) + 2
    idx2charpos = dict((idx, c) for c, idx in charpos2idx.items())

    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        desc_cut = jieba.lcut(desc)
        desc_pos = []
        for word in desc_cut:
            if len(word) == 1:
                desc_pos.append(word+'<S>')     # single character as one word
            else:
                for i in range(len(word)):
                    if i == 0:
                        desc_pos.append(word[i]+'<B>')  # begin
                    elif i == len(word) - 1:
                        desc_pos.append(word[i]+'<E>')  # end
                    else:
                        desc_pos.append(word[i]+'<M>')  # middle
        corpus.append(desc_pos)
    for data in tqdm(iter(train_data)):
        text_cut = jieba.lcut(data['text'])
        text_pos = []
        for word in text_cut:
            if len(word) == 1:
                text_pos.append(word + '<S>')  # single character as one word
            else:
                for i in range(len(word)):
                    if i == 0:
                        text_pos.append(word[i] + '<B>')  # begin
                    elif i == len(word) - 1:
                        text_pos.append(word[i] + '<E>')  # end
                    else:
                        text_pos.append(word[i] + '<M>')  # middle
        corpus.append(text_pos)
    return charpos2idx, idx2charpos, corpus


def train_valid_split(train_data):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)

    dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == 0]
    train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != 0]

    return train_data, dev_data


if __name__ == '__main__':
    # create directory
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    if not os.path.exists(SUBMIT_DIR):
        os.makedirs(SUBMIT_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    # load knowledge base data
    entity_desc, entity_id= load_kb_data(KB_FILENAME)
    mention_to_entity = load_m2e(M2E_FILENAME)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME), mention_to_entity)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_DESC_FILENAME), entity_desc)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, ENTITY_ID_FILENAME), entity_id)

    # load training data
    train_data = load_train_data(CCKS_TRAIN_FILENAME)

    # prepare word embedding
    word_vocab, idx2word, word_corpus = load_word_vocab_and_corpus(entity_desc, train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'), word_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='word'), idx2word)
    w2v = train_w2v(word_corpus, word_vocab)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='w2v'), w2v)

    # hold out split
    train_data, dev_data = train_valid_split(train_data)
    # load test data
    test_data = load_test_data(CCKS_TEST_FILENAME)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_FILENAME), train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_DATA_FILENAME), dev_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_DATA_FILENAME), test_data)
