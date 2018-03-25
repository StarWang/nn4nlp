import yaml
import torch
import random
import numpy as np
from trian import TriAN
from data_utils import set_seed, load_data, build_dict, get_acc, load_embedding
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
    optimizer = Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=0)
    lr_scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)
    loss_fn = BCELoss()

    if config['use_cuda']:
        model.cuda()
    model.train()

    # load embedding for word dictionary
    word_dict = w2i_lst[0]
    w2embed = load_embedding(word_dict, config['embedding_file'])
    for w, embedding in w2embed.items():
        model.embedding.weight.data[word_dict[w]].copy_(embedding)

    # save original embedding matrix. reset the embeddings for all vectors after topk every batch
    # this equals fine tuning topk embedding vectors every batch
    # questionable. seems like topk words in vocab are ranked by frequency.
    # why only fine tune vectors of stop words?
    # impact on performance to be investigated
    finetune_topk = config['finetune_topk']
    fixed_embedding = model.embedding.weight.data[finetune_topk].clone()

    for epoch in range(config['epoch']):
        train_data = random.shuffle(train_data)
        train_acc = []
        input_lst = ['d_words', 'd_pos', 'd_ner', 'd_mask', 'q_words', 'q_pos', 'q_mask',
                'c_words', 'c_mask', 'features', 'd_q_relation', 'd_c_relation']
        # training
        for batch_data in get_batches(train_data):
            model.embedding.weight.data[finetune_topk:] = fixed_embedding

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















