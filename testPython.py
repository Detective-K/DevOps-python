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

# NLP for NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from collections import Counter
from operator import itemgetter



# x is a string of json format
def cleanJson(x):
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


c_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}

# Compiling the contraction dict
c_re = re.compile('(%s)' % '|'.join(c_dict.keys()))

# List of stop words
add_stop = ['said', 'say', '...', 'like', 'cnn', 'ad']
stop_words = ENGLISH_STOP_WORDS.union(add_stop)

# List of punctuation
punc = list(set(string.punctuation))


# Splits words on white spaces (leaves contractions intact) and splits out
# trailing punctuation
def casual_tokenizer(text):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens


def expandContractions(text, c_re=c_re):
    def replace(match):
        return c_dict[match.group(0)]
    return c_re.sub(replace, text)


def process_text(text):
    text = casual_tokenizer(text)
    text = [each.lower() for each in text]
    text = [re.sub('[0-9]+', '', each) for each in text]
    text = [expandContractions(each, c_re=c_re) for each in text]
    text = [SnowballStemmer('english').stem(each) for each in text]
    text = [w for w in text if w not in punc]
    text = [w for w in text if w not in stop_words]
    text = [each for each in text if len(each) > 1]
    text = [each for each in text if ' ' not in each]
    return text

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
    # SOFfiledir = "D:\\Thesis\\Data\\SOF\\"
    # Quorafiledir = "D:\\Thesis\\Data\\Quora\\"
    SOFfiledir = "/Users/kko/Desktop/github/Thesis/Data/SOF/"
    Quorafiledir = "/Users/kko/Desktop/github/Thesis/Data/Quora/"
    #Get file data
    SOFexcels = [pd.read_excel(SOFfiledir+fname, engine='openpyxl') for fname in os.listdir(SOFfiledir) if 'xlsx' in fname]
    SOFdf = pd.concat(SOFexcels)
    SOFdf['activity.question.postCell'] = SOFdf['activity.question.postCell'].apply(lambda x: cleanJson(x))
    Quoraexcels = [pd.read_excel(Quorafiledir + fname, engine='openpyxl') for fname in os.listdir(Quorafiledir) if 'xlsx' in fname]
    Quoradf = pd.concat(Quoraexcels)
    #Merge stackoverflow and Quora data delete nan row
    Mergedf = {"MergeData":[] , "processed_text" : []}
    Mergedf = pd.DataFrame(Mergedf)
    Mergedf["MergeData"] = pd.concat([SOFdf['activity.question.postCell'], Quoradf['Answer']]).dropna(axis=0, how="any")

    # Data Preprocessing
    Mergedf["processed_text"] = Mergedf["MergeData"].apply(process_text)

    # # model = BERTopic(calculate_probabilities=True)
    model = BERTopic()
    topics, probabilities = model.fit_transform(Mergedf["MergeData"].to_list())

    bertopic0 = pd.DataFrame(model.get_topic(0))
    model.get_topic_info()
    # model.get_topic_freq().head()
    #
    #
    # #Visualize Topics (Add .show() for pycharm )
    # model.visualize_topics().show()

    # NMF Topic Modeling
    texts = Mergedf['processed_text']
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.85,
        max_features=50,
        ngram_range=(1, 2),
        preprocessor=' '.join
    )

    tfidf = tfidf_vectorizer.fit_transform(texts)

    nmf = NMF(
        n_components=20,
        init='nndsvd'
    ).fit(tfidf)

    # Get feature
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topic_words = nmf.components_
    document_topic_matrix = nmf.transform(tfidf)

    # print topic
    for topic_idx, topic in enumerate(topic_words):
        top_features = topic.argsort()[:-6:-1]
        print(f"Topic #{topic_idx + 1}:")
        for feature_index in top_features:
            print(feature_names[feature_index])
        print()

    # topic 分部率
    for doc_idx, doc in enumerate(Mergedf["processed_text"]):
        topic_probabilities = document_topic_matrix[doc_idx]
        print(f"Document #{doc_idx + 1}:")
        for topic_idx, prob in enumerate(topic_probabilities):
            print(f"Topic #{topic_idx + 1}: {prob:.2f}")
        print()



    return 0
if __name__ == "__main__": main()