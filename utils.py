import numpy as np
import jieba
import re
import json
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score
from gensim.models import Word2Vec
import pandas as pd

embedding_path = "models/word2vec.model"


def load_data(data_path, teacher_path):
    knowledge_path = teacher_path+"teacher_knowledge.json"
    train_path = data_path+'train.tsv'
    dev_path = data_path+'dev.tsv'
    test_path = data_path+'test.tsv'
    with open(knowledge_path, 'rb') as rd:
        knowledge = json.load(rd)
    train = pd.read_csv(train_path, sep='\t')
    dev = pd.read_csv(dev_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')
    train['text'] = train['text'].apply(lambda x: ' '.join(jieba.cut(x)))
    dev['text'] = dev['text'].apply(lambda x: ' '.join(jieba.cut(x)))
    test['text'] = test['text'].apply(lambda x: ' '.join(jieba.cut(x)))
    return train, dev, test, knowledge

# 长度分布统计
def analyse(data):
    t = data.apply(lambda x: len(x))
    print("max:{}, 0.99:{}, 0.95:{}, 0.90:{}, 0.85:{}".format(
        max(t), t.quantile(0.99), t.quantile(0.95), t.quantile(0.90), t.quantile(0.85)
    ))
# 读取文本数据
def data_process():
    pos_train = pd.read_excel('dataset/text.xlsx', header=None, sheet_name='广告数据')
    neg_train = pd.read_excel('dataset/text.xlsx', header=None, sheet_name='非广告数据')
    pos_train['label'] = 1
    neg_train['label'] = 0
    train = pd.concat([pos_train, neg_train])
    train.rename(columns={0: 'text'}, inplace=True)
    # train['text'] = train['text'].apply(lambda x: ' '.join(x))
    return train.sample(frac=1)

def to_tsv(data):
    data[0:3000].to_csv('dataset/train.tsv', sep='\t', index=False)
    data[3000:4000].to_csv('dataset/dev.tsv', sep='\t', index=False)
    data[4000:5000].to_csv('dataset/test.tsv', sep='\t', index=False)
    # with open('dataset/text.tsv', 'w', encoding='utf-8') as fout:
    #     fout.write()
    return data


def make_vocab(train, dev, test):
    vocab = dict()
    for article in [train['text'], dev['text'], test['text']]:
        for sentence in article:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = len(vocab) + 1
    return vocab


def w2v_embedding(train, dev, test, vocab):
    model = Word2Vec.load(embedding_path)
    model.train(train['text'], total_examples=model.corpus_count, epochs=model.iter)
    model.train(dev['text'], total_examples=model.corpus_count, epochs=model.iter)
    model.train(test['text'], total_examples=model.corpus_count, epochs=model.iter)

    embedding = np.zeros((len(vocab) + 1, 300))
    for word, token in vocab.items():
        try:
            embedding[token] = model.wv[word]
        except:
            continue
    return embedding


def seq_padding(x, vocab):
    max_len = 128
    data = []
    for i, sen in enumerate(x):
        t = []
        for word in sen:
            t.append(vocab[word])
        slen = len(sen)
        if slen < max_len:
            t = [0] * (max_len - slen) + t
            data.append(t)
        else:
            data.append(t[:int(max_len / 2)] + t[-int(max_len / 2):])
    return np.array(data)


def batch_iter(X, label, logit, feat10, feat11, feat12, vocab,
               batch_size=16, shuffle=True, seed=None):
    X, label = np.array(X), np.array(label)
    data_size = len(X)
    index = np.arange(data_size)
    num_batches_per_epoch = int((len(X) - 1) / batch_size) + 1

    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(index)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        idx = index[start_index: end_index]
        x_batch, y_batch, log_batch, feat10_batch, feat11_batch, feat12_batch = \
            X[idx], label[idx], logit[idx], feat10[idx], feat11[idx], feat12[idx]
        yield seq_padding(x_batch, vocab), y_batch, log_batch, \
              feat10_batch, feat11_batch, feat12_batch


def check(data):
    slen = np.zeros(len(data['x1_train']))
    for i, line in enumerate(data['x1_train']):
        slen[i] = len(data['x1_train'][i])
    print("0.9:{}, 0.95:{}, 0.99:{}, max:{}:".format(
        np.quantile(slen, 0.9), np.quantile(slen, 0.95), np.quantile(slen, 0.99), np.max(slen)))


def score(y_true, y_vote, y_pred=None):
    f1 = f1_score(y_true, y_vote, average='macro')
    acc = accuracy_score(y_true, y_vote)
    recall = recall_score(y_true, y_vote)
    if y_pred is not None:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = roc_auc_score(y_true, y_vote)
    return f1, acc, auc, recall