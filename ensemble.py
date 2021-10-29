# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: ensemble.py

@time: 2019/7/4 20:43

@desc:

"""

import os
import gc
import json

import jieba
import numpy as np
from keras import optimizers
import keras.backend as K
from config import ModelConfig, PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, MENTION_TO_ENTITY_FILENAME, \
    ENTITY_DESC_FILENAME, EMBEDDING_MATRIX_TEMPLATE, SUBMIT_DIR
from itertools import groupby
from models.recognition_model import RecognitionModel
from models.linking_model import LinkModel
from utils.data_loader import load_data
from utils.io import pickle_load, format_filename, submit_result, pickle_dump
from utils.other import pad_sequences_1d
from utils.other import pad_sequences_1d

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate, clipnorm=5)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))

def link(model_name, batch_size=2, n_epoch=1, learning_rate=0.001, optimizer_type='adam', embed_type=None,
         embed_trainable=True, use_relative_pos=False, n_neg=1, omit_one_cand=True, callbacks_to_add=None,
         swa_type=None, **kwargs):
    config = ModelConfig()
    config.model_name = model_name
    config.batch_size = batch_size
    print('batch_size')
    print(batch_size)
    config.n_epoch = n_epoch
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    config.embed_type = embed_type
    if embed_type:
        config.embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type=embed_type))
        config.embed_trainable = embed_trainable
    else:
        config.embeddings = None
        config.embed_trainable = True

    config.callbacks_to_add = callbacks_to_add or ['modelcheckpoint', 'earlystopping']

    config.vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'))
    config.vocab_size = len(config.vocab) + 2
    config.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))
    config.entity_desc = pickle_load(format_filename(PROCESSED_DATA_DIR, ENTITY_DESC_FILENAME))

    config.exp_name = '{}_{}_{}_{}_{}_{}'.format(model_name, embed_type if embed_type else 'random',
                                                 'tune' if embed_trainable else 'fix', batch_size, optimizer_type,
                                                 learning_rate)
    config.use_relative_pos = use_relative_pos
    if config.use_relative_pos:
        config.exp_name += '_rel'
    config.n_neg = n_neg
    if config.n_neg > 1:
        config.exp_name += '_neg_{}'.format(config.n_neg)
    config.omit_one_cand = omit_one_cand
    if not config.omit_one_cand:
        config.exp_name += '_not_omit'
    if kwargs:
        config.exp_name += '_' + '_'.join([str(k) + '_' + str(v) for k, v in kwargs.items()])
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str

    print('Logging Info - Experiment: %s' % config.exp_name)
    model = LinkModel(config, **kwargs)

    if swa_type is None:
        model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
        if not os.path.exists(model_save_path):
            raise FileNotFoundError('Linking model not exist: {}'.format(model_save_path))
        model.load_best_model()
    elif 'swa' in callbacks_to_add:
        model_save_path = os.path.join(config.checkpoint_dir, '{}_{}.hdf5'.format(config.exp_name,
                                                                                  swa_type))
        if not os.path.exists(model_save_path):
            raise FileNotFoundError('Linking model not exist: {}'.format(model_save_path))
        model.load_swa_model(swa_type)

    return model

def get_word_ids(word_cut):
    word_ids = []
    for word in word_cut:
        if config.vocab.get(word)==6 or config.vocab.get(word)==None:
            continue
        word_ids.append(config.vocab.get(word))  # all char in one word share the same word embedding
    return word_ids

def link_ensemble(link_model_list, text_data, test_mention, mention_begin, mention_to_entity, entity_desc, vocab, max_desc_len,
                  max_erl_len):
    print('Logging Info - Generate linking result:')
    pred_mention_entities = []
    for text, mention, _begin in zip(text_data, test_mention, mention_begin):
        print(text)
        _begin = int(_begin)
        pred_mention = [mention, _begin]
        print('cssssssss')
        print(pred_mention)
        link_result = []
        text_cut = jieba.lcut(text)
        text_ids = get_word_ids(text_cut)
        print(text_ids)
        cand_mention_entity = {}
        ent_owner = []
        ent_desc, begin, end, relative_pos = [], [], [], []
        if mention not in cand_mention_entity:  # there might be duplicated mention in pred_mentions
            cand_mention_entity[mention] = mention_to_entity.get(mention)

        _end = _begin
        _rel_pos = []
        for i in range(len(text_ids)):
            if i < _begin:
                _rel_pos.append(max(i - _begin + max_erl_len, 1))  # transform to id
            elif _begin <= i <= _end:
                _rel_pos.append(0 + max_erl_len)
            else:
                _rel_pos.append(min(i - _end + max_erl_len, 2 * max_erl_len - 1))
        for ent_id in cand_mention_entity[mention]:
            print('xxxxxxxxxxxxx')
            print(ent_id)
            desc = entity_desc[ent_id]
            desc_cut = jieba.lcut(desc)
            desc_ids = get_word_ids(desc_cut)
            ent_desc.append(desc_ids)
            begin.append([_begin])
            end.append([_end])
            relative_pos.append(_rel_pos)
            ent_owner.append(pred_mention)

        if ent_desc:
            cv_scores_list = []
            model = link_model_list
            model_inputs = []
            repeat_text_ids = np.repeat([text_ids], len(ent_desc), 0)
            model_inputs.append(repeat_text_ids)

            begin = np.array(begin)
            end = np.array(end)
            model_inputs.extend([begin, end])
            if model.config.use_relative_pos:
                relative_pos = pad_sequences_1d(relative_pos)
                model_inputs.append(relative_pos)
            ent_desc = pad_sequences_1d(ent_desc, max_desc_len)
            model_inputs.append(ent_desc)
            scores = model.link_model.predict(model_inputs)
            cv_scores_list.append(scores)

            ensemble_scores = np.mean(np.stack(cv_scores_list, axis=-1), axis=-1)[:, 0]
            for k, v in groupby(zip(ent_owner, ensemble_scores), key=lambda x: x[0]):
                score_to_rank = np.array([j[1] for j in v])
                ent_id = cand_mention_entity[k[0]][np.argmax(score_to_rank)]
                link_result.append((k[0], k[1], ent_id))

        pred_mention_entities.append(link_result)
    return pred_mention_entities

def link_evaluate(pred_mention_entities, gold_mention_entities):
    assert len(pred_mention_entities) == len(gold_mention_entities)
    match, n_true, n_pred = 1e-10, 1e-10, 1e-10
    for pred_mention_entity, gold_mention_entity in zip(pred_mention_entities, gold_mention_entities):
        pred = set(pred_mention_entity)
        true = set(gold_mention_entity)
        match += len(pred & true)
        n_pred += len(pred)
        n_true += len(true)

    r = match / n_true
    p = match / n_pred
    f1 = 2 * match / (n_pred + n_true)
    print('Logging Info - Recall: {}, Precision: {}, F1: {}'.format(r, p, f1))
    return r, p, f1


def rec_evaluate(data, pred_mentions):
    print('Logging Info - Evaluate over valid data:')
    match, n_true, n_pred = 1e-10, 1e-10, 1e-10
    for i in range(len(data)):
        pred = set(pred_mentions[i])
        true = set([(mention[0], mention[1]) for mention in data[i]['mention_data']])
        match += len(pred & true)
        n_pred += len(pred)
        n_true += len(true)
    r = match / n_true
    p = match / n_pred
    f1 = 2 * match / (n_pred + n_true)
    print('Logging Info - Recall: {}, Precision: {}, F1: {}'.format(r, p, f1))
    return r, p, f1


if __name__ == '__main__':
    predict_on_final_test = True
    if predict_on_final_test:
        test_data_type = 'test'
    else:
        test_data_type = 'test'

    '''entity linking model'''
    link_model = link('2step_el', batch_size=2, n_epoch=1, embed_type='w2v', embed_trainable=False,
                      score_func='cosine', margin=0.04, callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'],
                      n_neg=5, omit_one_cand=False, use_relative_pos=True, max_mention=True, add_cnn='after')

    el_group = list(range(1))
    '''model ensemble'''
    config = ModelConfig()
    config.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))
    config.vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'))
    test_data = load_data(test_data_type)
    test_text_data = [data['text'] for data in test_data]
    test_mention = [data['mention'] for data in test_data]
    mention_begin = [data['begin'] for data in test_data]

    config = link_model.config

    print('Loggin Info - EL group')
    ensemble_model_list = link_model

    print('Logging Info - Generate submission for test data:')
    test_pred_mention_entities = link_ensemble(ensemble_model_list, test_text_data, test_mention, mention_begin,
                                               config.mention_to_entity, config.entity_desc, config.vocab,
                                               config.max_desc_len, config.max_erl_len)
    test_submit_file = 'ensemble_el_{}_ensemble_{}submit.json'.format(el_group, 'final_' if predict_on_final_test else '')
    submit_result(test_submit_file, test_data, test_pred_mention_entities)

    '''results ensemble'''
    combine_file = 'final_submit.json'


