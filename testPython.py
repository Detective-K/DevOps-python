# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:39:26 2022

@author: d02044
"""
import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from torch import nn
import bz2 # 用于读取bz2压缩文件
from collections import Counter # 用于统计词频
import re # 正则表达式
import nltk # 文本预处理
import numpy as np

# def show_batch(imgs):
#    grid = utils.make_grid(imgs,nrow=5)
#    plt.imshow(grid.numpy().transpose((1, 2, 0)))
#    plt.title('Batch from dataloader')


# def pytorch1():
#     a = 1
#     img_data  = torchvision.datasets.ImageFolder( 'D:/Thesis/test',
#                                                 transform=transforms.Compose([
#                                                     transforms.Resize(256),
#                                                     transforms.CenterCrop(224),
#                                                     transforms.ToTensor()])
#                                                 )
    
#     print(len(img_data))
#     data_loader = torch.utils.data.DataLoader(img_data, batch_size=20,shuffle=True)
#     print(len(data_loader))    
    
#     for i, (batch_x, batch_y) in enumerate(data_loader):
#         if(i<4):
#             print(i, batch_x.size(), batch_y.size())
    
#             show_batch(batch_x)
#             plt.axis('off')
#             plt.show()
            
# def trya():
#     x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
#     y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
    
#     # 画图
#     plt.scatter(x.data.numpy(), y.data.numpy())
#     plt.show()


# class Tudui(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self,input):
#         ourput = input + 1
#         return ourput
          

# tudui = Tudui()
# x = torch.tensor(1.0)
# ourput = tudui(x)
# print(ourput)

        
# if __name__ =="__main__": trya()




train_file = bz2.BZ2File('D:/Testdata/train.ft.txt.bz2')
test_file = bz2.BZ2File('D:/Testdata/test.ft.txt.bz2')
train_file = train_file.readlines()
test_file = test_file.readlines()
print(train_file[0])

num_train = 800000
num_test = 200000

train_file = [x.decode('utf-8') for x in train_file[:num_train]]
test_file = [x.decode('utf-8') for x in test_file[:num_test]]

# 将__label__1编码为0（差评），__label__2编码为1（好评）
train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]

"""
`split(' ', 1)[1]`：将label和data分开后，获取data部分
`[:-1]`：去掉最后一个字符(\n)
`lower()`: 将其转换为小写，因为区分大小写对情感识别帮助不大，且会增加编码难度
"""
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])


for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])
        
for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])
        
words = Counter() # 用于统计每个单词出现的次数
for i, sentence in enumerate(train_sentences):
    words_list = nltk.word_tokenize(sentence) # 将句子进行分词
    words.update(words_list)  # 更新词频列表
    train_sentences[i] = words_list # 分词后的单词列表存在该列表中
    
    if i%200000 == 0: # 没20w打印一次进度
        print(str((i*100)/num_train) + "% done")
print("100% done")

words = {k:v for k,v in words.items() if v>1}
words = sorted(words, key=words.get,reverse=True)
print(words[:10]) # 打印一下出现次数最多的10个单词

