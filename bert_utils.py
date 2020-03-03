import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
vocab_path = '../input/roeberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# 读取文本数据
def load_data(data_path):
    train_path = data_path+'train.tsv'
    dev_path = data_path+'dev.tsv'
    test_path = data_path+'test.tsv'
    train = pd.read_csv(train_path, sep='\t')
    dev = pd.read_csv(dev_path, sep='\t')
    test = pd.read_csv(test_path, sep='\t')
    return train, dev, test

def analyse(data):
    t = data.apply(lambda x: len(x))
    print("max:{}, 0.99:{}, 0.95:{}, 0.90:{}, 0.85:{}".format(
        max(t), t.quantile(0.99), t.quantile(0.95), t.quantile(0.90), t.quantile(0.85)
    ))

def seq_padding(data, max_len=256, padding=0):
    for i in range(len(data)):
        t = []
        slen = len(data[i])
        if slen >= max_len:
            t = data[i][-int(max_len/2):]
            data[i] = data[i][:int(max_len/2)] + t
#             data[i] = data[i][:max_len]
        if slen < max_len:
            data[i] += [padding] * (max_len - slen)
    return data


class MyTokenizer(Tokenizer):
    def _tokenize(self, text):
        token = []
        for c in text:
            if c in self._token_dict:
                token.append(c)
            elif self._is_space(c):
                token.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                token.append('[UNK]')  # 剩余的字符是[UNK]
        return token


def token_dict():
    import codecs
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


vocab = token_dict()
tokenizer = MyTokenizer(vocab)
def batch_iter(s1, label, s2=None, max_len=512,
               batch_size=16, shuffle=True, seed=None):
    vocab = token_dict()
    tokenizer = MyTokenizer(vocab)
    s1, label = np.array(s1), np.array(label)
    if type(s2).__name__ != 'NoneType':
        s2 = np.array(s2)
    data_size = len(s1)
    index = np.arange(data_size)
    num_batches_per_epoch = int((len(s1) - 1) / batch_size) + 1

    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(index)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        idx = index[start_index: end_index]
        if type(s2).__name__ != 'NoneType':
            xs1, xs2 = sentence2token(s1[idx], s2[idx], max_len=max_len)
        else:
            xs1, xs2 = sentence2token(s1[idx], max_len=max_len)
        yield xs1, xs2, label[idx]


def sentence2token(s1, s2=None, max_len=512):
    xs1, xs2 = [], []
    for i in range(len(s1)):
        if type(s2).__name__ != 'NoneType':
            x1, x2 = tokenizer.encode(first=s1[i], second=s2[i])
        else:
            x1, x2 = tokenizer.encode(first=s1[i])
        xs1.append(x1)
        xs2.append(x2)
    return seq_padding(xs1, max_len=max_len), seq_padding(xs2, max_len=max_len)

def check(data):
    slen = np.zeros(len(data['text']))
    for i, line in enumerate(data['text']):
        slen[i] = len(data['text'][i])
    print("0.9:{}, 0.95:{}, 0.99:{}, max:{}:".format(
        np.quantile(slen, 0.9), np.quantile(slen, 0.95), np.quantile(slen,0.99), np.max(slen)))

def score(y_true, y_vote, y_pred=None):
    f1 = f1_score(y_true, y_vote, average='macro')
    acc = accuracy_score(y_true, y_vote)
    recall = recall_score(y_true, y_vote)
    if y_pred is not None:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = roc_auc_score(y_true, y_vote)
    return f1, acc, auc, recall