import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import LSTM_basic, QA_Dataset, LSTM_onebyone
import sys
import pickle
import numpy as np
import json

torch.manual_seed(11)
corpus_path = './corpus.pkl'
embedding_path = './pretrained_embed.npy'
save_model = False
embedding_dim = 300
hidden_dim = 32
batch_size = 32
learning_rate = 0.005
num_epochs = 30

def load_corpus(corpus_path):
    with open(corpus_path, 'rb') as handle:
        corpus = pickle.load(handle)
    return corpus

def load_embeddings(embed_path):
    return np.load(embed_path)

def save_checkpoint(state, fname):
    print("Save model to %s"%fname)
    torch.save(state, fname)

if __name__ == '__main__':
    corpus = load_corpus(corpus_path)
    embeddings = load_embeddings(embedding_path)
    # word2id = {y:x for x, y in corpus.items()}
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    model = LSTM_onebyone(embedding_dim = embedding_dim, hidden_dim = hidden_dim, vocab_size = len(corpus),\
                       batch_size = batch_size, use_cuda = use_cuda, embeddings = embeddings)
    print(model)
    if use_cuda:
        model = model.cuda()
    data_dir = './ready_data_onebyone'
    ftrain = 'mc160.train.tsv'
    fdev = 'mc160.dev.tsv'

    train_set = QA_Dataset(data_dir, ftrain, corpus, onebyone = True)
    dev_set = QA_Dataset(data_dir, fdev, corpus, onebyone = True)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    dev_loader = DataLoader(dev_set, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []

    rawQuestions, rawAnswers, rawLabels, rawOutput = [], [], [], []

    # Training 
    for epoch in range(num_epochs):
        total_acc = 0.0
        total_loss = 0.0
        total = 0
        for iter, data in enumerate(train_loader):
            questions, answers, labels, _, _, _ = data
            labels = torch.squeeze(labels)
            l = labels
            if use_cuda:
                q, l = Variable(questions.cuda()).t(), labels.cuda()
                a = Variable(answers[0].cuda()).t()
                b = Variable(answers[1].cuda()).t()
                c = Variable(answers[2].cuda()).t()
                d = Variable(answers[3].cuda()).t()
            else:
                q = Variable(questions).t()
                a = Variable(answers[0]).t()
                b = Variable(answers[1]).t()
                c = Variable(answers[2]).t()
                d = Variable(answers[3]).t()
            model.batch_size = len(labels)
            output = model(q, a, b, c, d)
            loss = loss_function(output.cpu(), Variable(l).cpu())
            loss.backward()
            optimizer.step()

            # Calculating training accuracy
            _, pred = torch.max(output.data, 1)
            total_acc += (pred.cpu() == labels).sum()
            total += len(labels)
            total_loss += loss.data[0]
        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)
        if save_model:
            fname = '{}/Epoch_{}.model'.format('./checkpoints', epoch)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                           fname = fname)

        # Testing/Validating
        total_acc = 0.0
        total_loss = 0.0
        total = 0
        for iter, data in enumerate(dev_loader):
            questions, answers, labels, origQuestions, origAnswers, origLabels = data
            labels = torch.squeeze(labels)
            l = labels
            if use_cuda:
                q, l = Variable(questions.cuda()).t(), labels.cuda()
                a = Variable(answers[0].cuda()).t()
                b = Variable(answers[1].cuda()).t()
                c = Variable(answers[2].cuda()).t()
                d = Variable(answers[3].cuda()).t()
            else:
                q = Variable(questions).t()
                a = Variable(answers[0]).t()
                b = Variable(answers[1]).t()
                c = Variable(answers[2]).t()
                d = Variable(answers[3]).t()

            model.batch_size = len(labels)
            output = model(q, a, b, c, d)
            loss = loss_function(output.cpu(), Variable(labels))
            if epoch == num_epochs - 1:
                rawQuestions.append(origQuestions)
                rawAnswers.append(origAnswers)
                rawLabels.append(origLabels)
                rawOutput.append(output.data.cpu().numpy())

            # Calculating training accuracy
            _, pred = torch.max(output.data, 1)
            total_acc += (pred.cpu() == labels).sum()
            total += len(labels)
            total_loss += loss.data[0]
        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)
        
        print("[Epoch %3d/%3d] Training loss: %.3f, Testing loss: %.3f, Training acc: %.3f, Testing acc: %.3f"%\
              (epoch, num_epochs, train_loss_[epoch], test_loss_[epoch],
                                 train_acc_[epoch], test_acc_[epoch]))

    correct, wrong = [], []
    for i in range(len(rawLabels)):
        for j in range(len(rawOutput[i])):
            # print(rawLabels[i][j], rawOutput[i][j])
            choice = int(np.argmax(rawOutput[i][j]))
            if rawLabels[i][j] == choice:
                correct.append((rawQuestions[i][j], rawAnswers[i][j], rawLabels[i][j], choice))
            else:
                wrong.append((rawQuestions[i][j], rawAnswers[i][j], rawLabels[i][j], choice))
    
    with open("correct_160.json", "w") as f:
        json.dump(correct, f)

    with open("wrong_160.json", "w") as f:
        json.dump(wrong, f)
                
