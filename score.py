# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : score.py
# @Software: PyCharm
# @Environment : Python 3.6+
# @Reference : https://github.com/likejazz/Siamese-LSTM

# 基础包
import pandas as pd
import jieba
import keras
from gensim.models import KeyedVectors
from util import make_w2v_embeddings, split_and_zero_padding, ManDist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

'''
本配置文件用于以服务形式调用预训练好的孪生网络预测新句对的相似度
'''

# ------------------预加载------------------ #

# 中英文训练选择，默认使用英文训练集
s = input("type cn or en:")
if s == 'cn':
    flag = 'cn'
    embedding_path = 'CnCorpus-vectors-negative64.bin'
    embedding_dim = 64
    max_seq_length = 20
    savepath = './data/cn_SiameseLSTM.h5'
else:
    flag = 'en'
    embedding_path = 'GoogleNews-vectors-negative300.bin'
    embedding_dim = 300
    max_seq_length = 10
    savepath = './data/en_SiameseLSTM.h5'

# 是否启用预训练的词向量，默认使用随机初始化的词向量
o = input("type yes or no for choosing pre-trained w2v or not:")
if o == 'yes':
    # 加载词向量
    print("Loading word2vec model(it may takes 2-3 mins) ...")
    embedding_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
else:
    embedding_dict = {}

# 加载预训练好的词向量和模型
model = keras.models.load_model(savepath, custom_objects={"ManDist": ManDist})
model.summary()


# -----------------主函数----------------- #

if __name__ == '__main__':

    while True:
        if flag == 'cn':
            # 输入待测试的句对
            sen1 = input("请输入句子1: ")
            sen2 = input("请输入句子2: ")
            dataframe = pd.DataFrame(
                {'question1': [" ".join(jieba.lcut(sen1))], 'question2': [" ".join(jieba.lcut(sen2))]})
        else:
            # 输入待测试的句对
            sen1 = input("input sentence1: ")
            sen2 = input("input sentence2: ")
            dataframe = pd.DataFrame({'question1': ["".join(sen1)], 'question2': ["".join(sen2)]})

        dataframe.to_csv("./data/test.csv", index=False, sep=',', encoding='utf-8')
        TEST_CSV = './data/test.csv'

        # 读取并加载测试集
        test_df = pd.read_csv(TEST_CSV)
        for q in ['question1', 'question2']:
            test_df[q + '_n'] = test_df[q]

        # 将测试集词向量化
        test_df, embeddings = make_w2v_embeddings(flag, embedding_dict, test_df, embedding_dim=embedding_dim)

        # 预处理
        X_test = split_and_zero_padding(test_df, max_seq_length)

        # 确认数据准备完毕且正确
        assert X_test['left'].shape == X_test['right'].shape

        # 预测并评估准确率
        prediction = model.predict([X_test['left'], X_test['right']])
        print(prediction)