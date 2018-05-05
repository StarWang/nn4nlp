import os
import time
import torch
import random
import numpy as np

from datetime import datetime

import pandas as pd
from utils import load_data, build_vocab, gen_submission, gen_final_submission
from get_sequence import getTitle, readScriptKnowledge
from config import args
from model import Model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(output_prefix):
    print (args)
    scriptKnowledge, _ = getTitle()
    trainScriptKnowledge = readScriptKnowledge('data/train_script.txt', scriptKnowledge)
    trialScriptKnowledge = readScriptKnowledge('data/trial_script.txt', scriptKnowledge)
    devScriptKnowledge = readScriptKnowledge('data/dev_script.txt', scriptKnowledge)
    testScriptKnowledge = readScriptKnowledge('data/test_script.txt', scriptKnowledge)

    build_vocab()
    train_data = load_data('./data/train-data-processed.json', trainScriptKnowledge,
                           args.use_script, args.use_char_emb)
    train_data += load_data('./data/trial-data-processed.json', trialScriptKnowledge,
                            args.use_script, args.use_char_emb)
    dev_data = load_data('./data/dev-data-processed.json', devScriptKnowledge,
                         args.use_script, args.use_char_emb)
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
            dev_acc = model.evaluate(dev_data, output_prefix)
            print('Dev accuracy: %f' % dev_acc)
        start_time = time.time()
        np.random.shuffle(train_data)
        cur_train_data = train_data

        model.train(cur_train_data)
        train_acc = model.evaluate(train_data[:2000], output_prefix, debug=False, eval_train=True)
        print('Train accuracy: %f' % train_acc)
        dev_acc = model.evaluate(dev_data, output_prefix, debug=True)
        print('Dev accuracy: %f' % dev_acc)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #os.system('mv ./data/output.log ./data/best-dev.log')
            model.save(checkpoint_path)
        elif args.test_mode:
            model.save(checkpoint_path)
        print('Best dev accuracy: %f' % best_dev_acc)
        print('Epoch %d use %d seconds.' % (i, time.time() - start_time))

    dev_data = load_data('./data/test-data-processed.json', testScriptKnowledge,
                         args.use_script, args.use_char_emb)
    model_path_list = [checkpoint_path]
    os.system('rm ./out-*.txt')
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
            for num_hop in [1, 2, 3]:
                for use_script in [True, False]:
                    for use_char_emb in [True, False]:
                        #char_emb_dim_lst = [50, 100, 150, 200] if use_char_emb else [-1]
                        char_emb_dim_lst = [50] if use_char_emb else [-1]
                        for char_emb_dim in char_emb_dim_lst:
                            set_seed(seed)
                            args.use_char_emb = use_char_emb
                            args.char_emb_dim = char_emb_dim
                            args.seed = seed
                            args.num_hop = num_hop
                            args.dropout_rnn_output = drop_out
                            args.dropout_emb = drop_out
                            args.use_script = use_script
                            args.pretrained = ''
                            # debugging
                            args.epoch = 2

                            accuracy = main('seed_{}_drop_out_{}_num_hop_{}_use_script_{}_use_char_emb_{}'
                                            'char_emb_dim_{}'.format(seed, drop_out, num_hop, use_script,
                                                                     use_char_emb, char_emb_dim))
                            stat.append([seed, drop_out, accuracy, num_hop, use_script, use_char_emb, char_emb_dim])
                            print ('-'*30 + 'STAT' + '-'*30)
                            print (pd.DataFrame(stat, columns=['seed', 'drop_out', 'dev_acc', 'num_hop', 'use_script',
                                                               'use_char_emb', 'char_emb_dim']))
                            print ('-'*30 + 'STAT' + '-'*30)

    np.save('statistics', stat)

