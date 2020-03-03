from keras_bert import AdamWarmup, calc_train_steps
from keras.layers import *
from keras import Model
from keras.optimizers import Adam
from bert_utils import *
import json
import gc


class Teacher_Bert:
    def __init__(self, train, test, dev, bert):
        self.train_data = train
        self.dev_data = dev
        self.test_data = test
        self.bert = bert

    def save_knowlege(self, x1_trn_tok, x2_trn_tok, model, knowledge_dict):
        def get_layer(layer_name):
            return model.get_layer(layer_name).output

        outputs = [get_layer('output_logit')]
        for i in range(10, 13):
            encoder_layer = get_layer('Encoder-{}-FeedForward-Norm'.format(i))
            cls_layer = Lambda(lambda x: x[:, 0])(encoder_layer)
            outputs.append(cls_layer)
        logit_layer = Model(inputs=model.input, outputs=outputs)
        feature = logit_layer.predict([x1_trn_tok, x2_trn_tok])
        knowledge_dict['logit'] = feature[0].tolist()
        for i in range(10, 13):
            knowledge_dict['layer_{}'.format(i)] = feature[i - 9].tolist()
        return knowledge_dict

    def create_model(self):
        cls = Lambda(lambda x: x[:, 0])(self.bert.output)  # 取出[CLS]对应的向量用来做分类
        dropout = Dropout(0.2)(cls)
        logit = Dense(2, name='output_logit')(dropout)
        output = Softmax(axis=-1)(logit)
        model = Model(self.bert.input, output)
        model.summary(line_length=200)
        gc.collect()
        return model

    def train(self):
        x_trn, y_trn = self.train_data['text'][:].values, self.train_data['label'][:].values
        x_val, y_val = self.dev_data['text'][:].values, self.dev_data['label'][:].values
        x_test, y_test = self.test_data['text'][:].values, self.test_data['label'][:].values
        folds, batch_size, steps, max_len = 5, 16, 30, 300
        y_vals = np.zeros((len(x_val), 2))
        y_vals_vote = np.zeros(len(x_val))
        y_test_pre = np.zeros((len(x_test), 2))
        knowledge_dict = dict()
        model = self.create_model()
        total_steps, warmup_steps = calc_train_steps(num_example=x_trn.shape[0],
                                                     batch_size=batch_size, epochs=steps, warmup_proportion=0.2)
        adamwarmup = AdamWarmup(total_steps, warmup_steps, lr=1e-5, min_lr=1e-7)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5))
        model.save_weights('origin')

        patient, best_score = 0, -1
        x1_trn_tok, x2_trn_tok = sentence2token(x_trn, max_len=max_len)
        x1_val_tok, x2_val_tok = sentence2token(x_val, max_len=max_len)
        for epoch in range(steps):
            # ==========train=========== #
            generator = batch_iter(x_trn, y_trn, max_len=max_len, batch_size=batch_size)
            for x1_tok, x2_tok, lab in generator:
                model.train_on_batch([x1_tok, x2_tok], np.eye(2)[lab])
            # ==========eval=========== #
            y_val_pre = model.predict([x1_val_tok, x2_val_tok])
            y_val_vote = np.argmax(y_val_pre, -1)  # 最大的值所在的索引作为预测结果
            f1, auc, acc, recall = score(y_val, y_val_vote)
            # ==========EarlyStop=========== #
            if f1 > best_score:
                patient = 0
                best_score = f1
                y_vals_vote = y_val_vote
                y_vals = y_val_pre
                model.save_weights('weight')
                # =========save knowledge==========
                knowledge_dict = self.save_knowlege(
                    x1_trn_tok, x2_trn_tok, model, knowledge_dict)

            print('epoch:{}, f1:{}, auc:{}, acc:{}, recall:{}, best_score:{}'.format(epoch, f1, auc, acc, recall,
                                                                                     best_score))
            patient += 1
            if patient >= 5:
                break
        # ==========加载最优模型预测测试集=========== #
        model.load_weights('weight')
        x1_test_tok, x2_test_tok = sentence2token(x_test, max_len=max_len)
        predict = np.argmax(model.predict([x1_test_tok, x2_test_tok]), -1)
        print('final dev score: ', score(y_val, y_vals_vote))
        print('final test score: ', score(y_test, predict))
        #         return y_test_vote, y_vals_vote, y_test, y_vals
        with open("teacher_knowledge.json", "w") as f:
            json.dump(knowledge_dict, f)