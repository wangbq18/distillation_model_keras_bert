Knowledge Distillation
----
代码是在kaggle的kernel上面写的（ipynb格式），迁移下来时还没来得及调试，直接运行可能会报错。  <br>
kernel的代码地址：https://www.kaggle.com/duolaaa/weibo-distil-student-layers3?scriptVersionId=29557259  <br>
代码实现了CNN、BiLSTM、Bert(3layers)对Bert(12layers)模型的蒸馏。有distillation、patient以及patient.full三种模式，分别蒸馏teacher的logit、output feature以及hiddent feature的知识。

### 参考文献
Patient Knowledge Distillation for BERT Model Compression <br>
Distilling the Knowledge in a Neural Network
