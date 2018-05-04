import torch
import numpy as np
import copy
import random
import re

from utils import vocab, pos_vocab, ner_vocab, rel_vocab

class Example:

    def __init__(self, input_dict, script_knowledge, use_script_knowledge):
        self.id = input_dict['id']
        self.passage_id = int(self.id.split('_')[-3])
        self.use_script_knowledge = use_script_knowledge
        self.passage = input_dict['d_words']
        self.question = input_dict['q_words']
        self.choice = input_dict['c_words']
        self.d_pos = input_dict['d_pos']
        self.d_ner = input_dict['d_ner']
        self.q_pos = input_dict['q_pos']
        assert len(self.q_pos) == len(self.question.split()), (self.q_pos, self.question)
        assert len(self.d_pos) == len(self.passage.split())
        self.features = np.stack([input_dict['in_q'], input_dict['in_c'], \
                                    input_dict['lemma_in_q'], input_dict['lemma_in_c'], \
                                    input_dict['tf']], 1)
        assert len(self.features) == len(self.passage.split())
        self.label = input_dict['label']
        self.script_knowledge_passage = []

        self.d_tensor = torch.LongTensor([vocab[w] for w in self.passage.split()])
        self.q_tensor = torch.LongTensor([vocab[w] for w in self.question.split()])
        self.c_tensor = torch.LongTensor([vocab[w] for w in self.choice.split()])
        self.d_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.d_pos])
        self.q_pos_tensor = torch.LongTensor([pos_vocab[w] for w in self.q_pos])
        self.d_ner_tensor = torch.LongTensor([ner_vocab[w] for w in self.d_ner])
        self.features = torch.from_numpy(self.features).type(torch.FloatTensor)
        self.p_q_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_q_relation']])
        self.p_c_relation = torch.LongTensor([rel_vocab[r] for r in input_dict['p_c_relation']])
        
        for sequence in script_knowledge[self.passage_id]:
            result = []
            for s in sequence:
                s = [vocab[w] for w in s.split(' ')]
                result.append(s)
            
            self.script_knowledge_passage.append(result)

        self._d_words_sentences = []
        passge_sentences = re.split(',|\.', input_dict['d_words'])
        for s in passge_sentences:
            s = [vocab[w] for w in s.split(' ')]
            self._d_words_sentences.append(s)

    def __str__(self):
        return 'Passage: %s\n Question: %s\n Answer: %s, Label: %d' % (self.passage, self.question, self.choice, self.label)
        
    @property
    def d_words_sentences(self):
        if len(self.script_knowledge_passage) == 0 or not self.use_script_knowledge:
            return self._d_words_sentences

        result = copy.deepcopy(self._d_words_sentences)
        index = random.randint(0, len(self.script_knowledge_passage)-1)
        result += self.script_knowledge_passage[index]
        
        return result

def _to_indices_and_mask(batch_tensor, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(0)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(1)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(0)
    if need_mask:
        return indices, mask
    else:
        return indices

def _to_feature_tensor(features):
    mx_len = max([f.size(0) for f in features])
    batch_size = len(features)
    f_dim = features[0].size(1)
    f_tensor = torch.FloatTensor(batch_size, mx_len, f_dim).fill_(0)
    for i, f in enumerate(features):
        f_tensor[i, :len(f), :].copy_(f)
    return f_tensor

def _pad_sentence_data(batch_seq):
    batch_size = len(batch_seq)
    max_sentence_len = max([len(seq) for seq in batch_seq])
    max_len = []
    for seq in batch_seq:
        for s in seq:
            max_len.append(len(s))
    max_len = max(max_len)

    # for word sequence, need a mask to extract the original words to when performing attention
    padded_batch = np.zeros((batch_size, max_sentence_len, max_len))
    for i, seq in enumerate(batch_seq):
        for j, s in enumerate(seq):
            padded_batch[i, j, :len(s)] = s

    return torch.LongTensor(padded_batch)

def batchify(batch_data):
    p, p_mask = _to_indices_and_mask([ex.d_tensor for ex in batch_data])
    p_pos = _to_indices_and_mask([ex.d_pos_tensor for ex in batch_data], need_mask=False)
    p_ner = _to_indices_and_mask([ex.d_ner_tensor for ex in batch_data], need_mask=False)
    p_q_relation = _to_indices_and_mask([ex.p_q_relation for ex in batch_data], need_mask=False)
    p_c_relation = _to_indices_and_mask([ex.p_c_relation for ex in batch_data], need_mask=False)
    q, q_mask = _to_indices_and_mask([ex.q_tensor for ex in batch_data])
    q_pos = _to_indices_and_mask([ex.q_pos_tensor for ex in batch_data], need_mask=False)
    choices = [ex.choice.split() for ex in batch_data]
    c, c_mask = _to_indices_and_mask([ex.c_tensor for ex in batch_data])
    f_tensor = _to_feature_tensor([ex.features for ex in batch_data])
    y = torch.FloatTensor([ex.label for ex in batch_data])
    p_sentences = _pad_sentence_data([ex.d_words_sentences for ex in batch_data])
    return p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, p_q_relation, p_c_relation, p_sentences, y
