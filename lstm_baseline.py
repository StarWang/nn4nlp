import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import LSTM_basic, QA_Dataset
import sys
import pickle
import numpy as np

torch.manual_seed(11)
corpus_path = './filtered_corpus.pickle'
embedding_path = './pretrained_embed.npy'
save_model = True
embedding_dim = 300
hidden_dim = 100
batch_size = 32
learning_rate = 0.05 
num_epochs = 50

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
    model = LSTM_basic(embedding_dim = embedding_dim, hidden_dim = hidden_dim, vocab_size = len(corpus),\
                       batch_size = batch_size, use_cuda = use_cuda, embeddings = embeddings)
    print(model)
    if use_cuda:
        model = model.cuda()
    data_dir = './ready_data_prepadding'
    ftrain = 'mc160.train.tsv'
    fdev = 'mc160.dev.tsv'

    train_set = QA_Dataset(data_dir, ftrain, corpus)
    dev_set = QA_Dataset(data_dir, fdev, corpus)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = False, num_workers = 4)
    dev_loader = DataLoader(dev_set, batch_size = batch_size, shuffle = False, num_workers = 4)
    
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []

    # Training 
    for epoch in range(num_epochs):
        total_acc = 0.0
        total_loss = 0.0
        total = 0
        for iter, data in enumerate(train_loader):
            questions, answers, labels = data
            labels = torch.squeeze(labels)
            l = labels
            if use_cuda:
                q, a, l = Variable(questions.cuda()).t(), Variable(answers.cuda()).t(), labels.cuda()
            else:
                q, a = Variable(questions).t(), Variable(answers).t()
            model.batch_size = len(labels)
            model.hidden = model.init_hidden()
            output = model(q, a)
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
            questions, answers, labels = data
            labels = torch.squeeze(labels)
            l = labels
            if use_cuda:
                q, a, l = Variable(questions.cuda()).t(), Variable(answers.cuda()).t(), labels.cuda()
            else:
                q, a = Variable(questions).t(), Variable(answers).t()

            model.batch_size = len(labels)
            model.hidden = model.init_hidden()
            output = model(q, a)
            loss = loss_function(output.cpu(), Variable(labels))

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

