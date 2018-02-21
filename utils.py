import torch
import torch.nn as nn
import json
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import os
import numpy as np

def data_loader(fpath):
    return json.load(open(fpath, 'r'))

class QA_Dataset(Dataset):
    def __init__(self, data_dir, qa_fname, corpus, maxlen = 128):
        self.data_dir = data_dir
        # Read all questions and answers as a tuple 
        fname = os.path.join(self.data_dir, qa_fname)
        qas = data_loader(fname)
        self.qas = qas
        self.corpus = corpus
        self.maxlen = maxlen
    
    def padding(self, lst):
        if len(lst) >= self.maxlen:
            return lst[:self.maxlen]
        pad_len = self.maxlen - len(lst)
        pads = [len(self.corpus) - 1] * pad_len
        lst += pads
        return lst

    def word2id(self, lst):
        return [self.corpus[word] if word in self.corpus else len(self.corpus) -1 for word in lst]

    def __getitem__(self, index):
        qa = self.qas[index]
        q = self.padding(self.word2id(qa['question']))
        a = self.padding(self.word2id(qa['answer']))
        l = qa['label']
        tq = torch.LongTensor(np.array(q, dtype = np.int64))
        ta = torch.LongTensor(np.array(a, dtype = np.int64))
        tl = torch.LongTensor([l])
        return tq, ta, tl

    def __len__(self):
        return len(self.qas)

class LSTM_basic(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, batch_size, use_cuda,\
                 embeddings = None, label_size = 4):
        super(LSTM_basic, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        if embeddings is not None:
            self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(embeddings))

            '''
            if use_cuda:
                self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(embeddings).cuda())
            else:
                self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(embeddings))
            # No fine-tune
            # self.word_embeddings.weight.requires_grad = False
            '''
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden2label = nn.Linear(hidden_dim * 2, label_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax()

    def init_hidden(self):
        '''
        if self.use_cuda:
            return Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),\
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            '''
        return Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),\
                    Variable(torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, question, answer):
        embed_q = self.word_embeddings(question)
        embed_a = self.word_embeddings(answer)

        out_q, hidden_q = self.lstm(embed_q)
        out_a, hidden_a = self.lstm(embed_a)
        qa = self.hidden2label(torch.cat([out_q[-1], out_a[-1]], 1))
        return self.softmax(qa)
