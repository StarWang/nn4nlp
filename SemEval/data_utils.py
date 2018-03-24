import torch
import random
import numpy as np
import json

class Sample():
    def __init__(self, info):
        # concatenation of dataset id (trial/train/dev/test), document id,
        # question id and choice id
        self.id = info['id']

        # text in the document, question and choice
        self.doc = info['d_words']
        self.question = info['q_words']
        self.choice = info['c_words']

        # label, for test set, the label is always -1
        self.label = info['label']

        ## features
        # for each word in the document, whether it appears in the question/choice
        self.in_q = info['in_q']
        self.in_c = info['in_c']

        # named entity for each document word
        self.d_ner = info['d_ner']

        # for each word in the document, whether its relation with each word in the question/choice indicated by ConceptNet
        self.d_q_relation = info['p_q_relation']
        self.d_c_relation = info['p_c_relation']

        # pos_tag of words in the passage/question
        self.d_pos = info['d_pos']
        self.q_pos = info['q_pos']

        # term frequency
        self.tf = info['tf']

        # lemma features (not mentioned in paper, no idea how it's crafted)
        self.lemma_in_q = info['lemma_in_q']
        self.lemma_in_c = info['lemma_in_c']



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [Sample(json.loads(sample.strip('\n'))) for sample in f]


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

def get_i2w(w2i):
    return
