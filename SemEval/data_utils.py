import torch
import random
import numpy as np
import json
import torch
import unicodedata
from torch.autograd import Variable
from torch import LongTensor, FloatTensor, ByteTensor
from collections import defaultdict

class Sample():
    # property: id, d_words, q_words, c_words, label, d_q_relation, d_c_relation, d_pos, q_pos, d_ner, features
    # next step: preprocess the document, question and choice to be integers
    def __init__(self, info, word_dict, pos_dict, ne_dict, relation_dict):
        # concatenation of dataset id (trial/train/dev/test), document id,
        # question id and choice id
        self.id = info['id']

        # text in the document, question and choice
        # normalize: characters are decomposed by canonical equivalence,
        # and multiple combining characters are arranged in a specific order
        self.d_words = [word_dict[normalize(w)] for w in info['d_words']]
        self.q_words = [word_dict[normalize(w)] for w in info['q_words']]
        self.c_words = [word_dict[normalize(w)] for w in info['c_words']]

        # label, for test set, the label is always -1
        self.label = info['label']

        ## features
        # for each word in the document, whether it appears in the question/choice
        in_q = info['in_q']
        in_c = info['in_c']

        # named entity for each document word
        self.d_ner = [ne_dict[w] for w in info['d_ner']]

        # for each word in the document, whether its relation with each word in the question/choice indicated by ConceptNet
        self.d_q_relation = [relation_dict[w] for w in info['p_q_relation']]
        self.d_c_relation = [relation_dict[w] for w in info['p_c_relation']]

        # pos_tag of words in the passage/question
        self.d_pos = [pos_dict[w] for w in info['d_pos']]
        self.q_pos = [pos_dict[w] for w in info['q_pos']]

        # term frequency
        tf = info['tf']

        # lemma features (not mentioned in paper, no idea how it's crafted)
        # guess is the lemmatized version of in_q/in_c
        lemma_in_q = info['lemma_in_q']
        lemma_in_c = info['lemma_in_c']

        # stack features
        self.features = [
                 in_q, in_c,
                 lemma_in_q, lemma_in_c,
                 tf
                ]

def build_dict(type):
    dct = defaultdict(lambda :len(dct))
    # NULL: used for padding
    dct['<NULL>'] = 0
    # UNK: unknown word
    dct['<UNK>'] = 1
    file_path = {'word':'./data/vocab', 'pos':'./data/pos_vocab', 'ne': './data/ner_vocab', 'relation':'./data/rel_vocab'}
    with open(file_path[type], 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip('\n')
            dct[normalize(word)]
    return dct

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_data(path, word_dict, pos_dict, ne_dict, relation_dict):
    with open(path, 'r', encoding='utf-8') as f:
        return [Sample(json.loads(sample.strip('\n')), word_dict, pos_dict, ne_dict, relation_dict) for sample in f]

def get_batches(data, batch_size):
    for i in range(len(data), batch_size):
        data_dct = pad_batch(data[i:i + batch_size])
        yield data_dct

def pad_batch_by_sequence(batch_seq, dtype, output_type=Variable):
    batch_size = len(batch_seq)
    max_len = max([len(seq) for seq in batch_seq])

    # for word sequence, need a mask to extract the original words to when performing attention
    mask = np.ones((batch_size, max_len))
    padded_batch = np.zeros((batch_size, max_len))
    for i, seq in enumerate(batch_seq):
        padded_batch[i, :len(seq)] = seq
        mask_batch[i, :len(seq)] = 0
    return output_type(dtype(padded_batch)), output_type(ByteTensor(mask))

def pad_batch_by_sequence_list(batch_seq_lst, dtype):
    batch_size =  len(batch_seq_lst)
    max_len = max([len(batch_seq_lst[i][0]) for i in range(batch_size)])
    feat_num = len(batch_seq_lst[0])
    result = []
    for i in range(len(batch_seq_lst[0])):
        result.append(pad_batch_by_sequence([batch_seq_lst[j][i] for j in range(batch_size)], dtype, lambda x:x)[0])
    return dtype(torch.cat(result, dim=1).resize_(batch_size, feat_num, max_len))

def pad_batch(batch_data):
    # sample property: id, d_words, q_words, c_words, label, d_q_relation, d_c_relation, d_pos, q_pos, d_ner, features
    # data to pad: word, pos, ne, relation, features
    q_words, q_mask = pad_batch_by_sequence([s.q_words for s in batch_data], LongTensor)
    d_words, d_mask = pad_batch_by_sequence([s.d_words for s in batch_data], LongTensor)
    c_words, c_mask = pad_batch_by_sequence([s.c_words for s in batch_data], LongTensor)

    d_q_relation, _ = pad_batch_by_sequence([s.d_q_relation for s in batch_data], LongTensor)
    d_c_relation, _ = pad_batch_by_sequence([s.d_c_relation for s in batch_data], LongTensor)

    q_pos, _ = pad_batch_by_sequence([s.q_pos for s in batch_data], LongTensor)
    d_pos, _ = pad_batch_by_sequence([s.d_pos for s in batch_data], LongTensor)

    d_ner, _ = pad_batch_by_sequence([s.d_ner for s in batch_data], LongTensor)

    features , _ = pad_batch_by_sequence_list([s.features for s in batch_data], FloatTensor)

    label = pad_batch_by_sequence([s.label for s in batch_data], FloatTensor)

    return {
            'q_words':q_words,
            'q_mask':q_mask,
            'd_words':d_words,
            'd_mask':d_mask,
            'd_q_relation':d_q_relation,
            'd_c_relation':d_c_relation,
            'q_pos':q_pos,
            'd_pos':d_pos,
            'd_ner':d_ner,
            'featuers':features,
            'label':label
            }

'''
def filter_embedding(embedding_path, word_dict, save_path):
    with open(embedding_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            info = line.split(' ')
            w = info[0]
            if w in word_dict:
            vec = [float(x) for x in info[1:]]

    pass
'''
def get_acc(y, pred):
    equal = y == (pred > 0.5)
    return equal.sum()/len(equal)

def get_i2w(w2i):
    return dict((v, k) for k, v in w2i.items())

def normalize(x):
    return unicodedata.normalize('NFD', x)

def load_embedding(word_dict, embedding_file_path):
    w2embed = defaultdict(list)
    w2i = get_i2w(word_dict)
    with open(embedding_file_path) as f:
        for line in f:
            line = line.strip('\n')
            info = line.split(' ')
            w = normalize(info[0])
            if w in word_dict:
                vec = [float(x) for x in info[1:]]
                w2embed[w].append(vec)
    for w, vec_lst in w2embed.items():
        w2embed[w] = np.mean(vec_lst)
    return w2embed
