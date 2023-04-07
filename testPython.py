# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:39:26 2022

@author: d02044
"""


# from bertopic import BERTopic
#
# from sklearn.datasets import fetch_20newsgroups
#
# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
#
# model = BERTopic()
# topics, probabilities = model.fit_transform(docs)
import json
from pandas import json_normalize
import pandas as pd



# x is a string of json format
def func(x):
    #s = json.loads(x)
    s = pd.json_normalize(eval(x))
    return s['comment'][0]

def main():
    #MLOPS取資料
    MLOPS_df = pd.read_excel("D:\Thesis\Data\SOF\MLOPS_SOF.xlsx", engine='openpyxl')
    MLOPS_df['activity.question.postCell'] = MLOPS_df['activity.question.postCell'].apply(lambda x: func(x))
    MLOPS_df = pd.read_excel("D:\Thesis\Data\SOF\MLOPS_SOF.xlsx", engine='openpyxl')
    MLOPS_df['activity.question.postCell'] = MLOPS_df['activity.question.postCell'].apply(lambda x: func(x))

    return 0
if __name__ == "__main__": main()