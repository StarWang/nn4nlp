import yaml
import torch
import random
import numpy as np
from trian import TriAN
from data_utils import set_seed, load_data, build_dict, get_acc
from torch.optim import Adamax
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR


if __name__ == '__main__':
    # load hyper parameters dictionary
    config = yaml.load(open('./config.yaml', 'r'))

    set_seed(config['seed'])

    # get word2ind dictionary, order: word, pos, ne, relation
    w2i_lst = []
    for t in ['word', 'pos', 'ne', 'relation']:
        w2i_lst.append(build_dict(t))

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

    # train_model
    model = TriAN(config)
    model.train()
    optimizer = Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=0)
    loss_fn = BCELoss()
    lr_scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    for epoch in range(config['epoch']):
        train_data = random.shuffle(train_data)
        train_acc = []
        input_lst = ['d_words', 'd_pos', 'd_ner', 'd_mask', 'q_words',
                'q_pos', 'q_mask', 'c_words', 'c_mask', 'features', 'd_q_relation', 'd_c_relation']
        # training
        for batch_data in get_batches(train_data):
            optimizer.zero_grad()
            y = batch_data['label']
            pred = model(*[batch_data[x] for x in input_lst])

            loss = loss_fn(pred, y)
            loss.backward()
            # clip grad norm to prevent gradient explosion
            torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clipping)
            optimizer.step()

            train_acc.append(get_acc(y.data.cpu().numpy(), pred.data.cpu().numpy()))
        lr_scheduler.step()
        print ('epoch:', epoch, 'training accuracy:', np.array(train_acc).mean())

        validation_acc = []
        # get accuracy in validation data
        for batch_data in get_batches(dev_data):
            y = batchc_data['label']
            pred = model(*[batch_data[x] for x in input_lst])
            loss = loss_fn(pred, y)

            validation_acc.append(get_acc(y.data.cpu().numpy(), pred.data.cpu().numpy()))
        print ('epoch:', epoch, 'validation accuracy:', np.array(validation_acc).mean())

    # save test prediction
    test_prediction = []
    for batch_data in get_batches(test_data):
        pred = model(*[batch_data[x] for x in input_lst])
        test_prediction += list(pred.data.cpu().numpy())
    with open('./data/test_output', 'w') as f:
        for prediction in test_prediction:
            f.write('{}\n'.format(int(prediction > 0.5)))
















