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



# x is a string of json format
def func(x):
    #s = json.loads(x)
    # s = pd.json_normalize(eval(x))
    # return s['comment'][0]
    return  ('' if pd.isna(x)  else pd.json_normalize(eval(x))['comment'][0])

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
    #Get file data
    SOFexcels = [pd.read_excel(SOFfiledir+fname, engine='openpyxl') for fname in os.listdir(SOFfiledir) if 'xlsx' in fname]
    SOFdf = pd.concat(SOFexcels)
    SOFdf['activity.question.postCell'] = SOFdf['activity.question.postCell'].apply(lambda x: func(x))
    Quoraexcels = [pd.read_excel(Quorafiledir + fname, engine='openpyxl') for fname in os.listdir(Quorafiledir) if 'xlsx' in fname]
    Quoradf = pd.concat(Quoraexcels)
    #Merge stackoverflow and Quora data delete nan row
    Mergedf = pd.concat([SOFdf['activity.question.postCell'], Quoradf['Answer']]).dropna(axis=0, how="any")

    model = BERTopic()
    topics, probabilities = model.fit_transform(Mergedf.to_list())
    #
    model.get_topic_info()
    model.get_topic_freq().head()


    #Visualize Topics (Add .show() for pycharm )
    model.visualize_topics().show()

    return 0
if __name__ == "__main__": main()