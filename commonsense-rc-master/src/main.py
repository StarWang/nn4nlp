import os
import time
import torch
import random
import numpy as np
import pandas as pd

from datetime import datetime

from utils import load_data, build_vocab, gen_submission, gen_final_submission

from config import args
from model import Model

def set_seed(seed):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def main(output_prefix):
    build_vocab()
    train_data = load_data('./data/train-data-processed.json')
    train_data += load_data('./data/trial-data-processed.json')
    dev_data = load_data('./data/dev-data-processed.json')
    if args.test_mode:
        # use validation data as training data
        train_data += dev_data
        dev_data = []
    model = Model(args)

    best_dev_acc = 0.0
    os.makedirs('./checkpoint', exist_ok=True)
    checkpoint_path = './checkpoint/%d-%s.mdl' % (args.seed, datetime.now().isoformat())
    print('Trained model will be saved to %s' % checkpoint_path)

    for i in range(args.epoch):
        print('Epoch %d...' % i)
        if i == 0:
            dev_acc = model.evaluate(dev_data)
            print('Dev accuracy: %f' % dev_acc)
        start_time = time.time()
        np.random.shuffle(train_data)
        cur_train_data = train_data

        model.train(cur_train_data)
        train_acc = model.evaluate(train_data[:2000], debug=False, eval_train=True)
        print('Train accuracy: %f' % train_acc)
        dev_acc = model.evaluate(dev_data, debug=True)
        print('Dev accuracy: %f' % dev_acc)
        print('Best dev accuracy: %f' % best_dev_acc)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            os.system('mv ./data/output.log ./data/best-dev.log')
            model.save(checkpoint_path)
        elif args.test_mode:
            model.save(checkpoint_path)
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    dev_data = load_data('./data/test-data-processed.json')
    model_path_list = [checkpoint_path]
    for model_path in model_path_list:
        print('Load model from %s...' % model_path)
        args.pretrained = model_path
        model = Model(args)

        # evaluate on development dataset
        dev_acc = model.evaluate(dev_data, output_prefix)
        print('dev accuracy: %f' % dev_acc)

        # generate submission zip file for Codalab
        prediction = model.predict(dev_data)
        gen_submission(dev_data, prediction)

    gen_final_submission(dev_data, output_prefix)
    print('Best dev accuracy: %f' % best_dev_acc)
    return best_dev_acc

if __name__ == '__main__':
    stat = []
    for seed in [1234, 123, 12, 1]:
        for drop_out in [0.4, 0.3, 0.2, 0.1]:
            args.seed = seed
            args.dropout_rnn_output = drop_out
            args.dropout_emb = drop_out
            accuracy = main('seed_{}_drop_out_{}'.format(seed, drop_out))
            stat.append([seed, drop_out, accuracy])
            print (pd.DataFrame(stat, columns=['seed', 'drop_out', 'dev_acc']))

    np.save(stat, 'statistics')


