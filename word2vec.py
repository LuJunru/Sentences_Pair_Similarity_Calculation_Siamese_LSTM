# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : train.py
# @Software: PyCharm
# @Environment : Python 3.6+
# @Reference : https://github.com/likejazz/Siamese-LSTM

# 基础包
import gensim
import logging

'''
本配置文件用于根据新语料训练词向量
'''

# ------------------预加载------------------ #

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# -----------------基础函数------------------ #

def extract_questions(filepath):  # 问题抽取
    final_list = []
    for line in open(filepath, 'r'):
        line_list = line.strip().split('\t')
        final_list += line_list[1:-1]
    return final_list


# -----------------主函数----------------- #

if __name__ == '__main__':

    documents = list(extract_questions("./data/atec_nlp_sim_train.csv"))  # 问题list
    logging.info("Done reading data file")
    model = gensim.models.Word2Vec(documents, size=60)
    model.train(documents, total_examples=len(documents), epochs=10)
    model.save("./data/atec_nlp_sim_train.w2v")