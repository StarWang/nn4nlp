from collections import defaultdict
import json
import os
from bs4 import BeautifulSoup
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

script_knowledge_path = './data/scripts'
transformer = TfidfTransformer(smooth_idf=False)
nlp = spacy.load('en')

def getTitle():
    result = defaultdict(list)
    wordDoc = defaultdict(list)
    for filename in os.listdir(script_knowledge_path):
        maxLength = 0
        longestItems = []
        key = filename.split('.')[0]
        with open(os.path.join(script_knowledge_path, filename)) as f:
            soup = BeautifulSoup(f, 'xml')
            scripts = soup.find_all('scripts')[0]
            for script in scripts.find_all('script'):
                items = script.find_all('item')
                if len(items) < 3:
                    continue
                for item in items:
                    if item.get('original'):
                        wordDoc[key].append(item.get('original'))
                    else:
                        wordDoc[key].append(item.get('text'))
                if len(items) > maxLength:
                    longestItems.append(items)
                    maxLength = len(items)
            
        for longItem in longestItems:
            sequence = []
            for item in longItem:
                text = ''
                if item.get('original'):
                    text = item.get('original')
                else:
                    text = item.get('text')
                tokens = nlp.tokenizer(text)
                sequence.append(' '.join([token.text for token in tokens]))
            
            result[key].append(sequence)
    
    result[''] = [['']]
    return result, wordDoc

def getCount(passages, num = 5):
    vectorizer = CountVectorizer(stop_words='english')
    corpus = vectorizer.fit_transform(passages)
    idx2words = {v: k for k, v in vectorizer.vocabulary_.items()}
    wordCount = corpus.toarray().sum(axis=0)
    inds = np.argpartition(wordCount, -num)[-num:]

    # return set([idx2words[i] for i in inds])
    return {idx2words[i]:wordCount[i] for i in inds}

def getTfIdf(passages, num = 5):
    result = []
    vectorizer = CountVectorizer(stop_words='english')
    corpus = vectorizer.fit_transform(passages)
    idx2words = {v: k for k, v in vectorizer.vocabulary_.items()}
    tfidf = transformer.fit_transform(corpus).toarray()

    for i in range(len(passages)):
        inds = np.argpartition(tfidf[i], -num)[-num:]
        words = [idx2words[i] for i in inds]
        result.append(set(words))

    return result

def getMatch(wordDoc, filepath, preprocessed_path):
    result = {}
    cleanFilenames = []

    wordDoc = list(wordDoc.items())
    filenames = list(map(lambda x: x[0], wordDoc))
    sequences = list(map(lambda x: x[1], wordDoc))

    for f in filenames:
        if '_' in f:
            key = f.split('_')
        else:
            key = f.split()

        cleanFilenames.append(key)

    sequenceResult = []
    for i in range(len(sequences)):
        count = getCount(sequences[i], num = 5)
        for word in cleanFilenames[i]:
            if word not in count:
                count[word] = min(count.values())
        sequenceResult.append(count)
    
    preprocessed = defaultdict(str)
    with open(preprocessed_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            passageId = data['id'].split('_')[-3]
            preprocessed[passageId] = data['d_words']
    
    passageIds = list(preprocessed.keys())
    passages = [preprocessed[k] for k in passageIds]

    vectorizer = CountVectorizer(stop_words='english')
    corpus = vectorizer.fit_transform(passages)
    idx2words = {v: k for k, v in vectorizer.vocabulary_.items()}
    for i, p in enumerate(corpus.toarray()):
        if 'vending' in passages[i]:
            result[passageIds[i]] = 'buy_from_vending_machine'
            continue
        elif 'tea' in passages[i]:
            result[passageIds[i]] = 'make_tea'
            continue
        elif 'grocery' in passages[i]:
            result[passageIds[i]] = 'going grocery shopping'
            continue
        elif 'dog' in passages[i] and 'walk' in passages[i]:
            result[passageIds[i]] = 'walk_the_dog'
            continue

        print(passages[i])
        inds = np.argpartition(p, -5)[-5:]
        countPassage = set([idx2words[i] for i in inds])
        keyIndex = sorted(range(len(sequenceResult)), key = lambda x : sum([sequenceResult[x].get(word, 0) for word in countPassage]), reverse=True)[:3]
        for j, key in enumerate(keyIndex):
            print(' '.join(cleanFilenames[key]))
            print((set(sequenceResult[key].keys()) & countPassage))
        
        newKeyIndex = sorted(keyIndex, key = lambda x: len(set(sequenceResult[x].keys()) & countPassage), reverse=True)
        key0, key1 = newKeyIndex[0], newKeyIndex[1]
        if len(set(sequenceResult[key0].keys()) & countPassage) == 0:
            result[passageIds[i]] = ''
            continue

        if len(set(sequenceResult[key0].keys()) & countPassage) == len(set(sequenceResult[key1].keys()) & countPassage):
            result[passageIds[i]] = filenames[keyIndex[0]]
            print(filenames[keyIndex[0]])
        else:
            result[passageIds[i]] = filenames[key0]
            print(filenames[key0])

        print()

    with open(filepath, 'w') as f:
        for passageId, filename in result.items():
            f.write(str(passageId) + ':' + filename + '\n')

def readScriptKnowledge(filepath, scriptKnowledge):
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            passageId, key = line.split(':')
            passageId = int(passageId)
            result[passageId] = scriptKnowledge[key.replace('\n', '')]
    
    return result

if __name__=="__main__":
    scriptKnowledge, wordDoc = getTitle()
    getMatch(wordDoc, 'data/trial_script.txt', 'data/trial-data-processed.json')
    getMatch(wordDoc, 'data/train_script.txt', 'data/train-data-processed.json')
    getMatch(wordDoc, 'data/dev_script.txt', 'data/dev-data-processed.json')
    getMatch(wordDoc, 'data/test_script.txt', 'data/test-data-processed.json')

