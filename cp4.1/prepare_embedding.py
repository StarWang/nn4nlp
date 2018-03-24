import gensim
import pickle
import numpy as np
model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

with open('corpus.pkl', 'rb') as handler:
    corpus = pickle.load(handler)

res = np.zeros(shape=(len(corpus), 300))
unknown = []
for k, v in corpus.items():
    if k in model:
        res[v] = model[k]
    else:
        unknown.append(k)
avg = np.mean(res, axis = 0)
for k in unknown:
    v = corpus[k]
    res[v] = avg

np.save('pretrained_embed', res)


with open('corpus_all.pkl', 'rb') as handler:
    corpus = pickle.load(handler)

res = np.zeros(shape=(len(corpus), 300))
unknown = []
for k, v in corpus.items():
    if k in model:
        res[v] = model[k]
    else:
        unknown.append(k)
avg = np.mean(res, axis = 0)
for k in unknown:
    v = corpus[k]
    res[v] = avg

np.save('pretrained_embed_all', res)

