import torch
import random
import numpy as np
import pandas as pd
import json
import torch
import unicodedata
from torch.autograd import Variable
from torch import LongTensor, FloatTensor, ByteTensor
from collections import defaultdict
import re
import copy

class Sample():
    # property: id, d_words, q_words, c_words, label, d_q_relation, d_c_relation, d_pos, q_pos, d_ner, features
    def __init__(self, info, word_dict, pos_dict, ne_dict, relation_dict, char_dict, script_knowledge):
        # concatenation of dataset id (trial/train/dev/test), document id,
        # question id and choice id
        self.id = info['id']
        self.passage_id = int(self.id.split('_')[-3])

        # normalize: characters are decomposed by canonical equivalence,
        # and multiple combining characters are arranged in a specific order
        self.d_words = [word_dict.get(normalize(w), 1) for w in info['d_words'].split(' ')]
        self.q_words = [word_dict.get(normalize(w), 1) for w in info['q_words'].split(' ')]
        self.c_words = [word_dict.get(normalize(w), 1) for w in info['c_words'].split(' ')]
        self.script_knowledge_passage = []

        self.d_chars = get_chars_ind_lst(char_dict, info['d_words'].split(' '))
        self.q_chars = get_chars_ind_lst(char_dict, info['q_words'].split(' '))
        self.c_chars = get_chars_ind_lst(char_dict, info['c_words'].split(' '))

        for sequence in script_knowledge[self.passage_id]:
            result = []
            for s in sequence:
                s = [word_dict.get(normalize(w), 1) for w in s.split(' ')]
                result.append(s)
            
            self.script_knowledge_passage.append(result)

        self._d_words_sentences = []
        passge_sentences = re.split(',|\.', info['d_words'])
        for s in passge_sentences:
            s = [word_dict.get(normalize(w), 1) for w in s.split(' ')]
            self._d_words_sentences.append(s)

        # text in the document, question and choice
        self.d_text = info['d_words']
        self.q_text = info['q_words']
        self.c_text = info['c_words']

        # label, for test set, the label is always -1
        self.label = info['label']

        ## features
        # for each word in the document, whether it appears in the question/choice
        in_q = info['in_q']
        in_c = info['in_c']

        # named entity for each document word
        self.d_ner = [ne_dict.get(w, 1) for w in info['d_ner']]

        # for each word in the document, whether its relation with each word in the question/choice indicated by ConceptNet
        self.d_q_relation = [relation_dict.get(w, 1) for w in info['p_q_relation']]
        self.d_c_relation = [relation_dict.get(w, 1) for w in info['p_c_relation']]

        # pos_tag of words in the passage/question
        self.d_pos = [pos_dict.get(w, 1) for w in info['d_pos']]
        self.q_pos = [pos_dict.get(w, 1) for w in info['q_pos']]

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
    
    @property
    def d_words_sentences(self):
        if len(self.script_knowledge_passage) == 0:
            return self._d_words_sentences

        result = copy.deepcopy(self._d_words_sentences)
        index = random.randint(0, len(self.script_knowledge_passage)-1)
        result += self.script_knowledge_passage[index]
        
        return result


def get_chars_ind_lst(char_dict, word_lst):
    chars = []
    for w in word_lst:
        chars.append([char_dict.get(normalize(c), 1) for c in w])
    return chars


def build_dict(type):
    dct = defaultdict(lambda :len(dct))
    # NULL: used for padding
    dct['<NULL>'] = 0
    # UNK: unknown word
    dct['<UNK>'] = 1
    file_path = {'word':'./data/vocab', 'pos':'./data/pos_vocab', 'ne': './data/ner_vocab',
                 'relation':'./data/rel_vocab', 'char':'./data/char_vocab'}
    with open(file_path[type], 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip('\n')
            dct[normalize(word)]
    return dct

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_data(path, word_dict, pos_dict, ne_dict, relation_dict, char_dict, script_knowledge):
    with open(path, 'r', encoding='utf-8') as f:
        return [Sample(json.loads(sample.strip('\n')), word_dict, pos_dict, ne_dict, relation_dict, char_dict, script_knowledge) for sample in f]

def get_batches(data, batch_size, use_cuda):
    for i in range(0, len(data), batch_size):
        data_dct = pad_batch(data[i:i + batch_size], use_cuda)
        yield data_dct

def pad_batch_by_sequence(batch_seq, dtype, use_cuda, output_type=Variable):
    batch_size = len(batch_seq)
    max_len = max([len(seq) for seq in batch_seq])

    # for word sequence, need a mask to extract the original words to when performing attention
    mask = np.ones((batch_size, max_len))
    padded_batch = np.zeros((batch_size, max_len))
    for i, seq in enumerate(batch_seq):
        padded_batch[i, :len(seq)] = seq
        mask[i, :len(seq)] = 0
    if use_cuda:
        return output_type(dtype(padded_batch).cuda()), output_type(ByteTensor(mask).cuda())
    else:
        return output_type(dtype(padded_batch)), output_type(ByteTensor(mask))

def pad_batch_by_sequence_list(batch_seq_lst, dtype, use_cuda):
    batch_size =  len(batch_seq_lst)
    max_len = max([len(batch_seq_lst[i][0]) for i in range(batch_size)])
    feat_num = len(batch_seq_lst[0])
    result = []
    for i in range(len(batch_seq_lst[0])):
        result.append(pad_batch_by_sequence([batch_seq_lst[j][i] for j in range(batch_size)], dtype, False, lambda x:x)[0])
    output = dtype(torch.cat(result, dim=1).resize_(batch_size, feat_num, max_len)).permute(0, 2, 1)
    if use_cuda:
        return Variable(output.cuda())
    return Variable(output)

def pad_sentence_data(batch_seq, dtype, use_cuda, output_type=Variable):
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
    if use_cuda:
        return output_type(dtype(padded_batch).cuda())
    else:
        return output_type(dtype(padded_batch))


# input format: [[w1c1, w1c2, ...], [w2c1, w2c2, ...], ...]
def pad_batch_by_char_seq(sent_word_char_lst, use_cuda):
    batch_size = len(sent_word_char_lst)
    max_sent_len = max([len(sent) for sent in sent_word_char_lst])
    max_word_len = max([len(word) for sent in sent_word_char_lst for word in sent])
    padded_lst = np.zeros((max_sent_len, batch_size, max_word_len))
    for i, sent in enumerate(sent_word_char_lst):
        for w_idx, w in enumerate(sent):
            padded_lst[w_idx, i, :] = w
    result = []
    for data in padded_lst:
        data = LongTensor(data).cuda() if use_cuda else LongTensor(data)
        result.append(Variable(data))
    # sent_len*batch_size*word_len
    return result


def pad_batch(batch_data, use_cuda):
    # sample property: id, d_words, q_words, c_words, label, d_q_relation, d_c_relation, d_pos, q_pos, d_ner, features
    # data to pad: word, pos, ne, relation, features
    q_words, q_mask = pad_batch_by_sequence([s.q_words for s in batch_data], LongTensor, use_cuda)
    d_words, d_mask = pad_batch_by_sequence([s.d_words for s in batch_data], LongTensor, use_cuda)
    c_words, c_mask = pad_batch_by_sequence([s.c_words for s in batch_data], LongTensor, use_cuda)

    q_chars = pad_batch_by_char_seq([s.q_chars for s in batch_data], use_cuda)
    d_chars = pad_batch_by_char_seq([s.d_chars for s in batch_data], use_cuda)
    c_chars = pad_batch_by_char_seq([s.c_chars for s in batch_data], use_cuda)

    d_q_relation, _ = pad_batch_by_sequence([s.d_q_relation for s in batch_data], LongTensor, use_cuda)
    d_c_relation, _ = pad_batch_by_sequence([s.d_c_relation for s in batch_data], LongTensor, use_cuda)

    q_pos, _ = pad_batch_by_sequence([s.q_pos for s in batch_data], LongTensor, use_cuda)
    d_pos, _ = pad_batch_by_sequence([s.d_pos for s in batch_data], LongTensor, use_cuda)

    d_ner, _ = pad_batch_by_sequence([s.d_ner for s in batch_data], LongTensor, use_cuda)

    features = pad_batch_by_sequence_list([s.features for s in batch_data], FloatTensor, use_cuda)
    d_words_sentences = pad_sentence_data([s.d_words_sentences for s in batch_data], LongTensor, use_cuda)

    ids = [s.id for s in batch_data]

    d_text = [s.d_text for s in batch_data]
    c_text = [s.c_text for s in batch_data]
    q_text = [s.q_text for s in batch_data]

    label = torch.FloatTensor([s.label for s in batch_data])
    if use_cuda:
        label = label.cuda()
    label = Variable(label)

    return {
            'q_words':q_words,
            'q_mask':q_mask,
            'q_text':q_text,
            'q_chars':q_chars,
            'd_words':d_words,
            'd_text':d_text,
            'd_chars':d_chars,
            'c_words':c_words,
            'c_mask':c_mask,
            'c_text':c_text,
            'c_chars':c_chars,
            'd_mask':d_mask,
            'd_q_relation':d_q_relation,
            'd_c_relation':d_c_relation,
            'q_pos':q_pos,
            'd_pos':d_pos,
            'd_ner':d_ner,
            'features':features,
            'label':label,
            'id':ids,
            'd_words_sentences': d_words_sentences
            }

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
    with open(embedding_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            info = line.split(' ')
            w = normalize(info[0])
            if w in word_dict:
                vec = [float(x) for x in info[1:]]
                w2embed[w].append(vec)
    for w, vec_lst in w2embed.items():
        w2embed[w] = np.mean(vec_lst, axis=0).astype('float32')
    print ('load embedding for {}/{} words'.format(len(w2embed), len(word_dict)))
    return w2embed


def predict(data, config, model, input_lst, error_analysis=False, evaluate=True):
    '''
     since we already know one question has one answer, we need to select the choice with
     the largest probability.
    '''
    pred_lst, y_lst = [], []
    id_lst, full_id_lst = [], []
    q_lst, d_lst, c_lst = [], [], []
    for batch_data in get_batches(data, config['batch_size'], config['use_cuda']):
        # id format: other_docid_qid_cid -> docid_qid
        id_lst += ['_'.join(x.split('_')[:3]) for x in batch_data['id']]
        full_id_lst += [x for x in batch_data['id']]
        y = batch_data['label']
        y_lst += y.data.cpu().numpy().tolist()
        pred = model(*[batch_data[x] for x in input_lst])
        pred_lst += pred.data.cpu().numpy().tolist()
        q_lst += batch_data['q_text']
        d_lst += batch_data['d_text']
        c_lst += batch_data['c_text']

    count, correct = 0, 0
    df = pd.DataFrame(np.stack([pred_lst, y_lst, id_lst, q_lst, d_lst, c_lst, full_id_lst], axis=1),
            columns=['pred', 'y', 'id', 'question', 'document', 'choice', 'full_id'])

    prediction_lst = []
    if error_analysis:
        error_file = open('error_records', 'w')
        correct_file = open('correct_records', 'w')

    for id, df_by_group in df.groupby('id'):
        count += 1
        prediction = df_by_group['pred'].values.argmax()
        y = df_by_group['y'].values.argmax()
        is_correct = prediction == y
        correct += is_correct

        prediction_lst.append(df_by_group['full_id'].iloc[prediction].split('_')[1:])

        if error_analysis:
            f = correct_file if is_correct else error_file
            case_num = correct if is_correct else count - correct
            f.write('case:{}\n'.format(case_num))
            f.write('<document>\n{}\n'.format(df_by_group['document'].iloc[0]))
            f.write('<question>\n{}\n'.format(df_by_group['question'].iloc[0]))
            f.write('<choices>\n')
            for i, choice in enumerate(df_by_group['choice']):
                f.write('{}:{}\n'.format(i, choice))
            f.write('pred:{}, truth:{}\n'.format(prediction, y))

    if error_analysis:
        error_file.close()
        correct_file.close()

    if evaluate:
        print ('max prob accuracy:{}/{}={}'.format(correct, count, correct/count))

    return prediction_lst, correct/count
