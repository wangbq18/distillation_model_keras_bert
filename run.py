from utils import *
from argument_parser import default_parser
from Teacher import Teacher_Bert
from distillation_model.DistilBiLSTM import StudentLSTM
from distillation_model.DistilCNN import StudentCNN
from distillation_model.DistilBert import StudentBert
from keras_bert import load_trained_model_from_checkpoint

if __name__ == '__main__':
    parser = default_parser()
    train, dev, test, teacher = load_data(parser.input_dir, parser.teacher_dir)
    # teacher
    config_path = 'models/chinese_roberta_wwm_ext_L-12_H-768_A-12/config.json'
    checkpoint_path = 'models/chinese_roberta_wwm_ext_L-12_H-768_A-12/model.ckpt'
    bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=300)
    Teacher_Bert(train, test, dev, bert)

    # distill
    if parser.task == 'DistilLSTM':
        vocab = make_vocab(train, dev, test)
        embedding = w2v_embedding(train, dev, test, vocab)
        model = StudentLSTM(
            train, dev, test, teacher, vocab, embedding, parser)
    elif parser.task == 'DistilCNN':
        vocab = make_vocab(train, dev, test)
        embedding = w2v_embedding(train, dev, test, vocab)
        model = StudentCNN(
            train, dev, test, teacher, vocab, embedding, parser)
    elif parser.task == 'DistilBert':
        for i, l in enumerate(bert.layers):
            if i < 8:
                l.trainable = False
            else:
                l.trainable = True
        model = StudentBert(
            train, dev, test, teacher, bert, parser)
    y_test, y_vals, y_vals_vote = model.train()