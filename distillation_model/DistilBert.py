from keras_bert import AdamWarmup, calc_train_steps
from keras.layers import *
from keras import Model
from keras import backend as K
from bert_utils import *


class StudentBert:
    def __init__(self, train, dev, test, teacher, bert, parse):
        self.train_data = train
        self.dev_data = dev
        self.test_data = test
        self.bert = bert
        self.teacher = teacher
        self.Mode = parse.Mode
        self.T = parse.T

    #         self.m = self.create_model()

    def distil_loss(self, soft=False, patient=False, T=5):
        def hard_target(y_true, y_pred):
            return K.categorical_crossentropy(y_true, y_pred)

        def soft_target(y_true, y_pred):
            y_true = K.softmax(y_true / T)
            y_pred = K.softmax(y_pred / T)
            return K.categorical_crossentropy(y_true, y_pred) * T ** 2

        if soft:
            return soft_target
        return hard_target

    def create_model(self):

        cls1 = Lambda(lambda x: x[:, 0])(
            self.bert.get_layer('Encoder-1-FeedForward-Norm').output)
        cls2 = Lambda(lambda x: x[:, 0])(
            self.bert.get_layer('Encoder-2-FeedForward-Norm').output)
        cls3 = Lambda(lambda x: x[:, 0])(
            self.bert.get_layer('Encoder-3-FeedForward-Norm').output)
        dropout = Dropout(0.2)(cls3)
        logit = Dense(2, name='output_logit')(dropout)
        output = Softmax(axis=-1)(logit)
        outputs = [output, logit]
        if self.Mode == 'patient':
            outputs.extend([cls3])
        elif self.Mode == 'patient.full':
            outputs.extend([cls1, cls2, cls3])
        else:
            raise ValueError('{} not recognized'.format(self.Mode))
        model = Model(self.bert.input, outputs=outputs)
        model.summary(line_length=200)
        return model

    def train(self):
        x_trn, y_trn = self.train_data['text'][:].values, self.train_data['label'][:].values
        x_val, y_val = self.dev_data['text'][:].values, self.dev_data['label'][:].values
        x_test, y_test = self.test_data['text'][:].values, self.test_data['label'][:].values
        folds, batch_size, steps, max_len = 5, 16, 30, 300
        y_vals_vote = np.zeros(len(y_val))
        best_score = 0
        model = self.create_model()
        total_steps, warmup_steps = calc_train_steps(num_example=x_trn.shape[0],
                                                     batch_size=batch_size, epochs=steps, warmup_proportion=0.2)
        adamwarmup = AdamWarmup(total_steps, warmup_steps, lr=1e-4, min_lr=1e-6)
        losses = [self.distil_loss(), self.distil_loss(soft=True, T=self.T)]
        if self.Mode == 'patient':
            losses.extend([self.distil_loss(soft=True, T=self.T)])
        elif self.Mode == 'patient.full':
            losses.extend([self.distil_loss(soft=True, T=self.T),
                           self.distil_loss(soft=True, T=self.T),
                           self.distil_loss(soft=True, T=self.T)])
        model.compile(loss=losses, optimizer=adamwarmup)

        x1_val_tok, x2_val_tok = sentence2token(x_val, max_len=max_len)
        knowledge = self.teacher
        logit, feature10, feature11, feature12 = np.array(knowledge['logit']), \
            np.array(knowledge['layer_10']), np.array(knowledge['layer_11']), np.array(knowledge['layer_12'])
        for epoch in range(steps):
            # ==========train=========== #
            generator = batch_iter(x_trn, y_trn, logit, feature10, feature11, feature12,
                                   max_len=max_len, batch_size=batch_size)
            for x1_tok, x2_tok, log, feat10, feat11, feat12, lab in generator:
                outputs = [np.eye(2)[lab], log]
                if self.Mode == 'patient':
                    outputs.extend([feat12])
                elif self.Mode == 'patient.full':
                    outputs.extend([feat10, feat11, feat12])
                model.train_on_batch(
                    [x1_tok, x2_tok], outputs)
            # ==========eval=========== #
            y_val_pre = model.predict([x1_val_tok, x2_val_tok])[0]
            y_val_vote = np.argmax(y_val_pre, -1)  # 最大的值所在的索引作为预测结果
            f1, auc, acc, recall = score(y_val, y_val_vote)
            # ==========EarlyStop=========== #
            if f1 > best_score:
                patient = 0
                best_score = f1
                y_vals_vote = y_val_vote
                model.save_weights('models/distil_bert_model')

            print('epoch:{}, f1:{}, auc:{}, acc:{}, recall:{}, best_score:{}'.format(
                epoch, f1, auc, acc, recall, best_score))
            patient += 1
            if patient >= 5:
                break
        # ==========加载最优模型预测测试集=========== #
        model.load_weights('models/distil_bert_model')
        x1_test_tok, x2_test_tok = sentence2token(x_test, max_len=max_len)
        predict = np.argmax(model.predict([x1_test_tok, x2_test_tok])[0], -1)
        print('final dev score: ', score(y_val, y_vals_vote))
        print('final test score: ', score(y_test, predict))

    def batch_iter(s1, label, logit, feat10, feat11, feat12, s2=None, max_len=512,
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
            yield xs1, xs2, logit[idx], feat10[idx], feat11[idx], feat12[idx], label[idx]