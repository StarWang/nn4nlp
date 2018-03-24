import yaml
import torch
import random
import numpy as np
from data_utils import set_seed, load_data, build_dict


if __name__ == '__main__':
    # load hyper parameters dictionary
    config = yaml.load(open('./config.yaml', 'r'))

    set_seed(config['seed'])

    # get word2ind dictionary, order: word, pos, ne, relation
    w2i_lst = []
    for t in ['word', 'pos', 'ne', 'relation']:
        w2i_lst.append(build_dict(t))

    # get word2ind/ind2word or feature2ind/ind2feature dictionary

    # load train data
    train_data = load_data('./data/train-data-processed.json', *w2i_lst)

    # load trial data
    trial_data = load_data('./data/trial-data-processed.json', *w2i_lst)

    # concatenate train data and trial data
    train_data += trail_data

    # load dev data
    dev_data = load_data('./data/dev-data-processed.json', *w2i_lst)

    # load test data
    test_data = load_data('./data/test-data-processed.json', *w2i_lst)

    for epoch in range(config['epoch']):
        train_data = random.shuffle(train_data)
        # get data in batch
        for batch_data in get_batches(train_data):
            f


        # train_model


        # calculate loss in dataset






