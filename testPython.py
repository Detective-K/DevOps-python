# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:39:26 2022

@author: d02044
"""


from bertopic import BERTopic

from sklearn.datasets import fetch_20newsgroups

# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
# model = BERTopic()
# topics, probabilities = model.fit_transform(docs)
import json
import os
from pandas import json_normalize
import pandas as pd


# NLP import
import re
import nltk
import nltk.corpus
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords


# NMF
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer




# x is a string of json format
def func(x):
    #s = json.loads(x)
    # s = pd.json_normalize(eval(x))
    # return s['comment'][0]
    return  ('' if pd.isna(x)  else pd.json_normalize(eval(x))['comment'][0])


def getCleanText(serie: str) -> str:
    stop_words = set(nltk.corpus.stopwords.words("english"))
    lem = WordNetLemmatizer()
    tokens = word_tokenize(str(serie))
    # tokens = [lem.lemmatize(t.lower()) for t in tokens if t not in stop_words and len(t) > 4]
    tokens = [lem.lemmatize(t.lower()) for t in tokens if t not in stop_words]
    cleaned = " ".join(tokens)
    return cleaned

# Get Clean text fun 2
def getCleanText2(serie: str) -> str:
    # Sentences
    sentences = nltk.sent_tokenize(str(serie))
    # Tokenize
    tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentences]
    # POS
    pos = [nltk.pos_tag(token) for token in tokens]
    # Lemmatization
    wordnet_pos = []
    for p in pos:
        for word, tag in p:
            if tag.startswith('J'):
                wordnet_pos.append(nltk.corpus.wordnet.ADJ)
            elif tag.startswith('V'):
                wordnet_pos.append(nltk.corpus.wordnet.VERB)
            elif tag.startswith('N'):
                wordnet_pos.append(nltk.corpus.wordnet.NOUN)
            elif tag.startswith('R'):
                wordnet_pos.append(nltk.corpus.wordnet.ADV)
            else:
                wordnet_pos.append(nltk.corpus.wordnet.NOUN)

    # Lemmatizer
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(p[n][0], pos=wordnet_pos[n]) for p in pos for n in range(len(p))]

    # Stopwords
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    tokens = [token for token in tokens if token not in nltk_stopwords]
    return tokens
    # # NER
    # ne_chunked_sents = [nltk.ne_chunk(tag) for tag in pos]
    # named_entities = []
    #
    # for ne_tagged_sentence in ne_chunked_sents:
    #     for tagged_tree in ne_tagged_sentence:
    #         if hasattr(tagged_tree, 'label'):
    #             entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
    #             entity_type = tagged_tree.label()
    #             named_entities.append((entity_name, entity_type))
    #             named_entities = list(set(named_entities))
    # return named_entities


def main():
    #取StackOverflow各項資料
    # MLOPS_df = pd.read_excel("D:\Thesis\Data\SOF\MLOPS_SOF.xlsx", engine='openpyxl')
    # MLOPS_df['activity.question.postCell'] = MLOPS_df['activity.question.postCell'].apply(lambda x: func(x))
    # DevOps_df = pd.read_excel("D:\Thesis\Data\SOF\DevOps_SOF.xlsx", engine='openpyxl')
    # DevOps_df['activity.question.postCell'] = DevOps_df['activity.question.postCell'].apply(lambda x: func(x))
    # AutomatedModelDeployment_df = pd.read_excel("D:\Thesis\Data\SOF\AutomatedModelDeployment_SOF.xlsx", engine='openpyxl')
    # DevOps_df['activity.question.postCell'] = DevOps_df['activity.question.postCell'].apply(lambda x: func(x))

    # pd.concat([MLOPS_df['activity.question.postCell'], DevOps_df['activity.question.postCell']])
    #Data root
    SOFfiledir = "D:\\Thesis\\Data\\SOF\\"
    Quorafiledir = "D:\\Thesis\\Data\\Quora\\"
    # SOFfiledir = "/Users/kko/Desktop/github/Thesis/Data/SOF/"
    # Quorafiledir = "/Users/kko/Desktop/github/Thesis/Data/Quora/"
    #Get file data
    SOFexcels = [pd.read_excel(SOFfiledir+fname, engine='openpyxl') for fname in os.listdir(SOFfiledir) if 'xlsx' in fname]
    SOFdf = pd.concat(SOFexcels)
    SOFdf['activity.question.postCell'] = SOFdf['activity.question.postCell'].apply(lambda x: func(x))
    Quoraexcels = [pd.read_excel(Quorafiledir + fname, engine='openpyxl') for fname in os.listdir(Quorafiledir) if 'xlsx' in fname]
    Quoradf = pd.concat(Quoraexcels)
    #Merge stackoverflow and Quora data delete nan row
    Mergedf = {"MergeData":[] , "processed_text" : []}
    Mergedf = pd.DataFrame(Mergedf)
    Mergedf["MergeData"] = pd.concat([SOFdf['activity.question.postCell'], Quoradf['Answer']]).dropna(axis=0, how="any")

    # Data Preprocessing
    Mergedf["processed_text"] = Mergedf["MergeData"].apply(getCleanText2)

    # model = BERTopic(calculate_probabilities=True)
    model = BERTopic()
    topics, probabilities = model.fit_transform(Mergedf.to_list())
    #
    model.get_topic_info()
    model.get_topic_freq().head()


    #Visualize Topics (Add .show() for pycharm )
    model.visualize_topics().show()

    # 假设我们有一些文本数据存储在一个列表中
    documents = [
        "这是一篇关于机器学习的文章",
        "主题模型可以帮助我们发现文本数据中的隐藏主题",
        "机器学习是人工智能领域的重要技术之一"
    ]

    # 创建一个TF-IDF向量化器
    # vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.85
    )

    # 将文本数据转换为TF-IDF特征矩阵
    tfidf_matrix = vectorizer.fit_transform(Mergedf["processed_text"].to_list())

    # 定义NMF模型并拟合数据
    num_topics = 2  # 假设我们希望得到2个主题
    nmf_model = NMF(n_components=num_topics)
    nmf_model.fit(tfidf_matrix)

    # 获取主题词和文档-主题矩阵
    feature_names = vectorizer.get_feature_names_out()
    topic_words = nmf_model.components_
    document_topic_matrix = nmf_model.transform(tfidf_matrix)

    # 打印每个主题的关键词
    for topic_idx, topic in enumerate(topic_words):
        top_features = topic.argsort()[:-6:-1]  # 获取每个主题中的前5个关键词
        print(f"Topic #{topic_idx + 1}:")
        for feature_index in top_features:
            print(feature_names[feature_index])
        print()

    # 打印每个文档的主题分布
    for doc_idx, doc in enumerate(documents):
        topic_probabilities = document_topic_matrix[doc_idx]
        print(f"Document #{doc_idx + 1}:")
        for topic_idx, prob in enumerate(topic_probabilities):
            print(f"Topic #{topic_idx + 1}: {prob:.2f}")
        print()

    return 0
if __name__ == "__main__": main()