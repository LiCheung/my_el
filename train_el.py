# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train_el.py.py

@time: 2019/5/17 11:03

@desc:

"""


import os
import gc
import time
import numpy as np
from keras import backend as K
from keras import optimizers
from config import ModelConfig, PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, MENTION_TO_ENTITY_FILENAME, \
    ENTITY_DESC_FILENAME, ENTITY_ID_FILENAME, EMBEDDING_MATRIX_TEMPLATE, LOG_DIR, PERFORMANCE_LOG
from models.linking_model import LinkModel
from utils.data_loader import LinkDataGenerator, load_data
from utils.io import pickle_load, format_filename, write_log

from utils.io import pickle_dump, format_filename
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

def train_link(model_name, batch_size=32, n_epoch=1, learning_rate=0.001, optimizer_type='adam',
               embed_type=None, embed_trainable=True, callbacks_to_add=None,
               use_relative_pos=False, n_neg=1, omit_one_cand=True, overwrite=True, swa_start=5,
               early_stopping_patience=3, **kwargs):
    config = ModelConfig()
    config.model_name = model_name
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    config.embed_type = embed_type
    if embed_type:
        config.embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type=embed_type),allow_pickle=True)
        config.embed_trainable = embed_trainable
    else:
        config.embeddings = None
        config.embed_trainable = True

    config.callbacks_to_add = callbacks_to_add or ['modelcheckpoint', 'earlystopping']
    if 'swa' in config.callbacks_to_add:
        config.swa_start = swa_start
        config.early_stopping_patience = early_stopping_patience

    config.vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'))
    config.vocab_size = len(config.vocab) + 2
    config.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))
    config.entity_desc = pickle_load(format_filename(PROCESSED_DATA_DIR, ENTITY_DESC_FILENAME))

    config.entity_id = pickle_load(format_filename(PROCESSED_DATA_DIR, ENTITY_ID_FILENAME))

    config.exp_name = '{}_{}_{}_{}_{}_{}'.format(model_name, embed_type if embed_type else 'random',
                                                 'tune' if config.embed_trainable else 'fix', batch_size, optimizer_type,
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

    # logger to log output of training process
    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type, 'epoch': n_epoch,
                 'learning_rate': learning_rate, 'other_params': kwargs}

    print('Logging Info - Experiment: %s' % config.exp_name)
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = LinkModel(config, **kwargs)

    train_data_type, dev_data_type = 'train', 'dev'
    train_generator = LinkDataGenerator(train_data_type, config.vocab, config.mention_to_entity, config.entity_desc,config.entity_id,
                                        config.batch_size, config.max_desc_len, config.max_erl_len,
                                        config.use_relative_pos, config.n_neg, config.omit_one_cand)

    dev_data = load_data(dev_data_type)
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.train(train_generator, dev_data)
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    model.load_best_model()
    dev_text_data, dev_pred_mentions, dev_gold_mention_entities = [], [], []
    for data in dev_data:
        dev_text_data.append(data['text'])
        dev_pred_mentions.append(data['mention_data'])
        dev_gold_mention_entities.append(data['mention_data'])
    # for data in dev_data:
    #     mention_data = []
    #     dev_text_data.append(data['text'])
    #     #data['mention_data'][2] = config.entity_id(data['mention_data'])
    #     print(data['mention_data'][0])
    #     mention = data['mention_data'][0]
    #     print(mention[2])
    #     m_entity_id = config.entity_id[mention[2]]
    #     mention_data.append((mention[0],mention[1],m_entity_id))
    #     dev_pred_mentions.append(mention_data)
    #     dev_gold_mention_entities.append(mention_data)
    print('Logging Info - Evaluate over valid data:')
    r, p, f1 = model.evaluate(dev_text_data, dev_pred_mentions, dev_gold_mention_entities)
    train_log['dev_performance'] = (r, p, f1)

    swa_type = None
    if 'swa' in config.callbacks_to_add:
        swa_type = 'swa'
    elif 'swa_clr' in config.callbacks_to_add:
        swa_type = 'swa_clr'
    if swa_type:
        model.load_swa_model(swa_type)
        print('Logging Info - Evaluate over valid data based on swa model:')
        r, p, f1 = model.evaluate(dev_text_data, dev_pred_mentions, dev_gold_mention_entities)
        train_log['swa_dev_performance'] = (r, p, f1)

    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG, model_type='2step_el'), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()


if __name__ == '__main__':
    train_link('2step_el', batch_size=8, n_epoch=1, embed_type='w2v', embed_trainable=False,
               callbacks_to_add=['swa', 'modelcheckpoint', 'earlystopping'], n_neg=2, omit_one_cand=False,
               use_relative_pos=True, score_func='exponential', margin=0.04, max_mention=True, add_cnn='after')