import yaml
import torch
import random
import numpy as np
from trian import TriAN
from data_utils import set_seed, load_data, build_dict, get_acc, load_embedding, get_batches, predict
from torch.optim import Adamax
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR


if __name__ == '__main__':
    # load hyper parameters dictionary
    config = yaml.load(open('./config.yaml', 'r'))
    config['use_cuda'] = config['use_cuda'] and torch.cuda.is_available()

    set_seed(config['seed'])

    # get word2ind dictionary, order: word, pos, ne, relation
    w2i_lst = []
    for t in ['word', 'pos', 'ne', 'relation']:
        w2i_lst.append(build_dict(t))

    # load train data
    print ('loading training data')
    train_data = load_data('./data/train-data-processed.json', *w2i_lst)
    print ('train size:', len(train_data))
    #print (np.array(train_data[-1].id))
    #np.save(r'C:\Users\WANG\Desktop\reimplementation\nn4nlp\SemEval\re_data',
            #np.array([np.array(s.c_words) for s in train_data]))
    #raise
    #print (np.array(train_data[-1].d_pos))
    #print (np.array(train_data[-1].q_pos))
    #print (np.array(train_data[-1].d_ner))
    #raise

    # load trial data
    print ('loading trial data')
    trial_data = load_data('./data/trial-data-processed.json', *w2i_lst)
    print ('trial size:', len(trial_data))

    # concatenate train data and trial data
    train_data += trial_data

    # load dev data
    print ('loading validation data')
    dev_data = load_data('./data/dev-data-processed.json', *w2i_lst)
    print ('validation size:', len(dev_data))

    # load test data
    print ('loading test data')
    test_data = load_data('./data/test-data-processed.json', *w2i_lst)
    print ('test size:', len(test_data))

    # train_model
    print ('creating model')
    model = TriAN(config, [len(dct) for dct in w2i_lst])
    optimizer = Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=0)
    lr_scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    # could try maximizing margin loss next time
    loss_fn = BCELoss()

    if config['use_cuda']:
        model.cuda()
    model.train()

    # load embedding for word dictionary
    print ('loading embeddings')
    word_dict = w2i_lst[0]
    w2embed = load_embedding(word_dict, config['embedding_file'])
    for w, embedding in w2embed.items():
        model.embeddings.wordEmbedding.weight.data[word_dict[w]].copy_(torch.from_numpy(embedding.astype('float32')))

    # save original embedding matrix. reset the embeddings for all vectors after topk every batch
    # this equals fine tuning topk embedding vectors every batch
    # questionable. seems like topk words in vocab are ranked by frequency.
    # why only fine tune vectors of stop words?
    # impact on performance to be investigated
    finetune_topk = config['finetune_topk']
    fixed_embedding = model.embeddings.wordEmbedding.weight.data[finetune_topk:].clone()
    np.save(r'C:\Users\WANG\Desktop\reimplementation\nn4nlp\SemEval\re_embed', model.embeddings.wordEmbedding.weight.data.numpy())

    input_lst = ['d_words', 'd_pos', 'd_ner', 'd_mask', 'q_words', 'q_pos', 'q_mask',
                'c_words', 'c_mask', 'features', 'd_q_relation', 'd_c_relation']

    print ('start training')
    validation_acc = []
    # get accuracy in validation data
    for batch_data in get_batches(dev_data, config['batch_size'], config['use_cuda']):
        y = batch_data['label']
        pred = model(*[batch_data[x] for x in input_lst])
        loss = loss_fn(pred, y)

        validation_acc.append(get_acc(y.data.cpu().numpy(), pred.data.cpu().numpy()))
    print ('epoch:', 0, 'validation accuracy binary:', np.array(validation_acc).mean())
    predict(dev_data, config, model, input_lst)

    for epoch in range(config['epoch']):
        model.train()
        random.shuffle(train_data)
        train_acc = []

        # training
        for batch_data in get_batches(train_data, config['batch_size'], config['use_cuda']):
            model.embeddings.wordEmbedding.weight.data[finetune_topk:] = fixed_embedding

            optimizer.zero_grad()
            y = batch_data['label']
            pred = model(*[batch_data[x] for x in input_lst])

            loss = loss_fn(pred, y)
            loss.backward()

            # clip grad norm to prevent gradient explosion
            torch.nn.utils.clip_grad_norm(model.parameters(), config['grad_clipping'])
            optimizer.step()

            train_acc.append(get_acc(y.data.cpu().numpy(), pred.data.cpu().numpy()))
            if len(train_acc) % 50 == 0:
                print('{} th batches, loss: {}'.format(len(train_acc), loss.data[0]))
        lr_scheduler.step()
        print ('lr:', lr_scheduler.get_lr()[0])
        print ('epoch:', epoch, 'training accuracy binary:', np.array(train_acc).mean())
        predict(train_data, config, model, input_lst)

        validation_acc = []
        # get accuracy in validation data
        model.eval()
        for batch_data in get_batches(dev_data, config['batch_size'], config['use_cuda']):
            y = batch_data['label']
            pred = model(*[batch_data[x] for x in input_lst])
            loss = loss_fn(pred, y)

            validation_acc.append(get_acc(y.data.cpu().numpy(), pred.data.cpu().numpy()))
        print ('epoch:', epoch, 'validation accuracy binary:', np.array(validation_acc).mean())
        predict(dev_data, config, model, input_lst)

    predict(dev_data, config, model, input_lst, error_analysis=True, evaluate=True)

    # save test prediction
    with open('./data/test_output', 'w') as f:
        for prediction in predict(test_data, config, model, input_lst, error_analysis=False, evaluate=False):
            f.write('{}\n'.format(prediction))

