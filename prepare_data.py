import json
import os, sys
sys.path.append('./')
import pickle
from utils import replica

fnames = os.listdir('./data/')
fnames = [fname for fname in fnames if '.tsv' in fname]

fpath = './' + 'ready_data_all/'
corpus = set()
padding = 'UNK'
def pad_ans(lst, threshold):
    if len(lst) >= threshold:
        return lst[:threshold]
    pads = [padding] * (threshold - len(lst))
    return lst + pads

for fname in fnames:
    xs = []
    print("Read from %s"%(os.path.join('./data/', fname)))
    label_fname = os.path.join('./data/', fname).replace('tsv', 'ans')
    with open(label_fname, 'r') as f:
        ys = f.readlines()
    ys = [y.strip().split() for y in ys]
    ys = list(sum(ys, []))
    print("    Read its labels %s containing %d labels"%(label_fname, len(ys)))
    tot_cnt = 0
    for line in open(os.path.join('./data/', fname), 'r'):
        x = json.loads(line)
        pid = x['id']
        passage = x['passage']
        passage = pad_ans(passage, 96)
        qas = x['questions']
        corpus.update(passage)
        cnt = 0
        for qa in qas:
            q = qa['tokens']
            a = qa['answers']
            a = [replica(x, 32) for x in a]
            q = pad_ans(q, 32)
            alist = list(sum(a, []))
            data = {}
            data['question'] =list(sum([passage, q], []))
            data['answer'] = alist
            data['pid'] = pid
            data['qid'] = cnt
            data['label'] = 0 if ys[tot_cnt] == 'A' else 1 if ys[tot_cnt] == 'B'\
                    else 2 if ys[tot_cnt] == 'C' else 3
            tot_cnt += 1
            cnt += 1
            corpus.update(alist)
            corpus.update(q)
            xs.append(data)
    with open(os.path.join(fpath, fname), 'w') as outfile:
        json.dump(xs, outfile)
corpus.add('UNK')
lst = list(corpus)
sorted(lst)
word2id = {k:v for v, k in enumerate(lst)}
with open('corpus_all.pkl', 'wb') as handle:
    pickle.dump(word2id, handle)
