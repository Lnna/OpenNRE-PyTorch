# 两个项目之间的api调用
import sys
sys.path.append('/home/nana/Documents/pycharmforlinux/mParser')
from src.gen_mediate_para import hs_parse
print(hs_parse([('NN','人们'),('VV','爱'),('NN','小明')]))
# import numpy as np
#
# a=np.zeros((20,300))
# b=np.zeros((20,300))
#
# print(np.stack([a,b]).shape)

#修改开源库代码的正确方式：继承覆盖
# import logging
# from stanfordcorenlp.corenlp import StanfordCoreNLP
#
# class StanfordNlp(StanfordCoreNLP):
#     def __init__(self, path_or_host, port=None, memory='4g', lang='en', timeout=1500, quiet=True,
#                  logging_level=logging.WARNING):
#         super(StanfordNlp,self).__init__(path_or_host,lang=lang)
#     def pos_tag(self, sentence):
#         r_dict = self._request('pos', sentence)
#         words = []
#         tags = []
#         for s in r_dict['sentences']:
#             for token in s['tokens']:
#                 words.append(token['word'])
#                 tags.append(token['pos'])
#         return list(zip(words, tags))
# nlp = StanfordNlp(r'/home/nana/Documents/stanford-corenlp-full-2016-10-31/', lang='zh')
# print(nlp.pos_tag('真是一个好日子'))




import dynet as dy
# define the parameters
m = dy.ParameterCollection()
W = m.add_parameters((8,2))
V = m.add_parameters((1,8))
b = m.add_parameters((8))
print(b.npvalue())
# renew the computation graph
dy.renew_cg()

# create the network
x = dy.vecInput(2) # an input vector of size 2.
output = dy.logistic(V*(dy.tanh((W*x)+b)))
# define the loss with respect to an output y.
y = dy.scalarInput(0) # this will hold the correct answer
loss = dy.binary_log_loss(output, y)

# create training instances
def create_xor_instances(num_rounds=2000):
    questions = []
    answers = []
    for round in range(num_rounds):
        for x1 in 0,1:
            for x2 in 0,1:
                answer = 0 if x1==x2 else 1
                questions.append((x1,x2))
                answers.append(answer)
    return questions, answers

questions, answers = create_xor_instances()

# train the network
trainer = dy.SimpleSGDTrainer(m)

total_loss = 0
seen_instances = 0
for question, answer in zip(questions, answers):
    x.set(question)
    y.set(answer)
    seen_instances += 1
    total_loss += loss.value()
    loss.backward()
    trainer.update()
    if (seen_instances > 1 and seen_instances % 100 == 0):
        print("average loss is:",total_loss / seen_instances)




