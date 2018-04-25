import yaml
import torch
import random
import sys
sys.path.extend("./")
sys.path.extend("./../")
import copy
import numpy as np
from trian import TriAN
from data_utils import set_seed, load_data, build_dict, get_acc, load_embedding, get_batches, predict
from torch.optim import Adamax
from torch.nn import BCELoss
from torch.optim.lr_scheduler import MultiStepLR
from get_sequence import getTitle, readScriptKnowledge

def saveModel(model, path):
    state_dict = copy.copy(model.state_dict())
    params = {'state_dict': state_dict}
    torch.save(params, path)

def loadModel(model, path):
    saved_params = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = saved_params['state_dict']
    model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    # load hyper parameters dictionary
    config = yaml.load(open('./config.yaml', 'r'))
    config['use_cuda'] = config['use_cuda'] and torch.cuda.is_available()

    set_seed(config['seed'])

    # get word2ind dictionary, order: word, pos, ne, relation
    w2i_lst = []
    for t in ['word', 'pos', 'ne', 'relation']:
        w2i_lst.append(build_dict(t))

    scriptKnowledge, _ = getTitle()
    trainScriptKnowledge = readScriptKnowledge('data/train_script.txt', scriptKnowledge)
    trialScriptKnowledge = readScriptKnowledge('data/trial_script.txt', scriptKnowledge)
    devScriptKnowledge = readScriptKnowledge('data/dev_script.txt', scriptKnowledge)
    testScriptKnowledge = readScriptKnowledge('data/test_script.txt', scriptKnowledge)

    # load train data
    print ('loading training data')
    train_data = load_data('./data/train-data-processed.json', *w2i_lst, trainScriptKnowledge)
    print ('train size:', len(train_data))
    
    # load trial data
    print ('loading trial data')
    trial_data = load_data('./data/trial-data-processed.json', *w2i_lst, trialScriptKnowledge)
    print ('trial size:', len(trial_data))

    # concatenate train data and trial data
    train_data += trial_data

    # load dev data
    print ('loading validation data')
    dev_data = load_data('./data/dev-data-processed.json', *w2i_lst, devScriptKnowledge)
    print ('validation size:', len(dev_data))

    # load test data
    print ('loading test data')
    test_data = load_data('./data/test-data-processed.json', *w2i_lst, testScriptKnowledge)
    print ('test size:', len(test_data))

    # train_model
    print ('creating model')
    model = TriAN(config, [len(dct) for dct in w2i_lst])
    optimizer = Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=0)
    lr_scheduler = MultiStepLR(optimizer, milestones=[3, 6, 10, 13], gamma=0.5)

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

    input_lst = ['d_words', 'd_pos', 'd_ner', 'd_mask', 'q_words', 'q_pos', 'q_mask',
                'c_words', 'c_mask', 'features', 'd_q_relation', 'd_c_relation', 'd_words_sentences']

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

    bestAccy = -1
    if config['pretrained'] is not None:
        model = loadModel(model, config['pretrained'])

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
        # predict(train_data, config, model, input_lst)

        validation_acc = []
        # get accuracy in validation data
        model.eval()
        for batch_data in get_batches(dev_data, config['batch_size'], config['use_cuda']):
            y = batch_data['label']
            pred = model(*[batch_data[x] for x in input_lst])
            loss = loss_fn(pred, y)

            validation_acc.append(get_acc(y.data.cpu().numpy(), pred.data.cpu().numpy()))
        print ('epoch:', epoch, 'validation accuracy binary:', np.array(validation_acc).mean())
        _, accy = predict(dev_data, config, model, input_lst)
        if(accy > bestAccy):
            saveModel(model, 'data/best_model')
            accy = bestAccy
        saveModel(model, 'checkpoint/model_%d'%epoch)
    predict(dev_data, config, model, input_lst, error_analysis=True, evaluate=True)

    # save test prediction
    with open('./data/answer' + config['ensemble_index'] + '.txt', 'w') as f:
        model = loadModel(model, 'data/best_model')
        predictions, _ = predict(test_data, config, model, input_lst, error_analysis=False, evaluate=True)
        predictions = sorted(predictions)
        for prediction in predictions:
            f.write(','.join(prediction) + '\n')

