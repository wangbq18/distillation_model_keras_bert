from keras.layers import *
from keras import Model
from keras import backend as K
from keras.optimizers import Adam
from utils import *
import gc


class StudentLSTM:
    def __init__(self, train, dev, test, teacher, vocab, embedding, parse):
        self.train_data = train
        self.dev_data = dev
        self.test_data = test
        self.embedding = embedding
        self.vocab = vocab
        self.teacher = teacher
        self.Mode = parse.Mode
        self.T = parse.T

    def distil_loss(self, soft=False, patient=False, T=1):
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
        embedding = self.embedding
        # (batch_size, seq_len)
        inputs = Input(shape=(None,))
        emb_layer = Embedding(embedding.shape[0], 768, trainable=True)
        emb = emb_layer(inputs)  # (batch_size, seq_len, dim)
        emb = Dropout(0.2)(emb)
        bilstm1 = Bidirectional(CuDNNLSTM(384, return_sequences=True))(emb)
        bilstm2 = Bidirectional(CuDNNLSTM(384, return_sequences=True))(bilstm1)
        poolings1 = GlobalMaxPooling1D()(bilstm1)
        poolings2 = GlobalMaxPooling1D()(bilstm2)
        poolings_norm1 = BatchNormalization()(poolings1)
        poolings_norm2 = BatchNormalization()(poolings2)

        logit = Dense(units=2)(poolings2)
        output = Softmax(axis=-1)(logit)

        if self.Mode == 'distil':
            model = Model(inputs=[inputs], outputs=[
                output, logit])
        elif self.Mode == 'patient':
            model = Model(inputs=[inputs], outputs=[
                output, logit, poolings_norm2])
        elif self.Mode == 'patient.full':
            model = Model(inputs=[inputs], outputs=[
                output, logit, poolings_norm1, poolings_norm2])
        else:
            raise ValueError('{} not recognized'.format(self.Mode))
        model.summary()
        return model

    def train(self):
        x_trn, y_trn = self.train_data['text'].values, self.train_data['label'].values
        x_val, y_val = self.dev_data['text'].values, self.dev_data['label'].values
        x_test, y_test = self.test_data['text'].values, self.test_data['label'].values
        folds, batch_size, steps = 5, 256, 100
        y_vals_vote = np.zeros(len(y_val))
        model = self.create_model()
        losses = [self.distil_loss(), self.distil_loss(soft=True, T=self.T)]
        if self.Mode == 'patient':
            losses.extend([self.distil_loss(soft=True, T=self.T)])
        elif self.Mode == 'patient.full':
            losses.extend([self.distil_loss(soft=True, T=self.T),
                           self.distil_loss(soft=True, T=self.T)])
        model.compile(loss=losses, optimizer=Adam(lr=1e-3))
        patient, best_score = 0, 0
        knowledge = self.teacher
        logit, feature10 = np.array(knowledge['logit']), np.array(knowledge['layer_10'])
        feature11, feature12 = np.array(knowledge['layer_11']), np.array(knowledge['layer_12'])
        for epoch in range(steps):
            generator = batch_iter(x_trn, y_trn, logit, feature10, feature11, feature12, self.vocab)
            for x_batch, y_batch, log_batch, feat10, feat11, feat12 in generator:
                outputs = [np.eye(2)[y_batch], log_batch]
                if self.Mode == 'patient':
                    outputs.extend([feat12])
                elif self.Mode == 'patient.full':
                    outputs.extend([feat10, feat12])
                model.train_on_batch([x_batch], outputs)
            x_val_tok = seq_padding(x_val, self.vocab)
            y_val_pre = model.predict(x_val_tok)[0]
            y_val_vote = np.argmax(y_val_pre, -1)  # 最大的值所在的索引作为预测结果
            f1, auc, acc, recall = score(y_val, y_val_vote)
            # ==========EarlyStop=========== #
            if f1 > best_score:
                patient = 0
                best_score = f1
                y_vals_vote = y_val_vote
                y_vals = y_val_pre
                model.save_weights('models/distil_lstm_model')
            print('epoch:{}, f1:{}, auc:{}, acc:{}, recall:{}, best_score:{}'.format(
                epoch, f1, auc, acc, recall, best_score))
            patient += 1
            if patient >= 10:
                break
        # ==========加载最优模型预测测试集=========== #
        model.load_weights('models/distil_lstm_model')
        test_tok = seq_padding(x_test, self.vocab)
        predict = np.argmax(model.predict([test_tok])[0], -1)
        gc.collect()
        print('final dev score: ', score(y_val, y_vals_vote))
        print('final test score: ', score(y_test, predict))