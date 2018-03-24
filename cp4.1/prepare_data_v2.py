import json
import os, sys
sys.path.append('./')
import pickle

fnames = os.listdir('./data/')
fnames = [fname for fname in fnames if '.tsv' in fname]

fpath = './' + 'ready_data_128_fillup/'
corpus = set()
padding = 'UNK'
def pad_ans(lst, threshold):
    if len(lst) >= threshold:
        return lst[:threshold]
    pads = [padding] * (threshold - len(lst))
    return lst + pads

def replica(lst, threshold):
    if len(lst) >= threshold:
        return lst[:threshold]
    replicas = lst * (int(threshold / len(lst)) + 1)
    return replicas[:threshold]

def check_label(idx , ans):
    if idx == 0 and ans == 'A':
        return 1
    if idx == 1 and ans == 'B':
        return 1
    if idx == 2 and ans == 'C':
        return 1
    if idx == 3 and ans == 'D':
        return 1
    return 0

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
        passage = replica(passage, 96)
        qas = x['questions']
        corpus.update(passage)
        cnt = 0
        for qa in qas:
            q = qa['tokens']
            a = qa['answers']
            a = [replica(x, 128) for x in a]
            q = replica(q, 32)
            cur_label = 0
            for cur_a in a:
                data = {}
                data['question'] =list(sum([passage, q], []))
                data['answer'] = cur_a
                data['pid'] = pid
                data['qid'] = cnt
                data['label'] = check_label(cur_label, ys[tot_cnt])
                cur_label += 1
                corpus.update(cur_a)
            tot_cnt += 1
            cnt += 1
            corpus.update(q)
            xs.append(data)
    with open(os.path.join(fpath, fname), 'w') as outfile:
        json.dump(xs, outfile)

corpus.add('UNK')
lst = list(corpus)

sorted(lst)
word2id = {k:v for v, k in enumerate(lst)}
with open('corpus.pkl', 'wb') as handle:
    pickle.dump(word2id, handle)
