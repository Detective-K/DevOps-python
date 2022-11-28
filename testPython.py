# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:39:26 2022

@author: d02044
"""
import torch
from torch.utils.data import TensorDataset, DataLoader
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

words = ['_PAD'] + words
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

for i, sentence in enumerate(train_sentences):    
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i, sentence in enumerate(test_sentences):
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

def pad_input(sentences, seq_len):
    """
    将句子长度固定为`seq_len`，超出长度的从后面阶段，长度不足的在前面补0
    """
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# 固定测试数据集和训练数据集的句子长度
train_sentences = pad_input(train_sentences, 200)
test_sentences = pad_input(test_sentences, 200)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

#--------------------------------------以上為數據前處理

#-----------------------------------建模開始
batch_size = 200

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

#設定使用GPU還是CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


#模型建構
class SentimentNet(nn.Module):
    def __init__(self, vocab_size):
        super(SentimentNet, self).__init__()
        self.n_layers = n_layers = 2 # LSTM的层数
        self.hidden_dim = hidden_dim = 512 # 隐状态的维度，即LSTM输出的隐状态的维度为512
        embedding_dim = 400 # 将单词编码成400维的向量
        drop_prob=0.5 # dropout
        
        # 定义embedding，负责将数字编码成向量，详情可参考：javascript:void(0)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, # 输入的维度
                            hidden_dim, # LSTM输出的hidden_state的维度
                            n_layers, # LSTM的层数
                            dropout=drop_prob, 
                            batch_first=True # 第一个维度是否是batch_size
                           )
        
        
        
        # LSTM结束后的全连接线性层
        self.fc = nn.Linear(in_features=hidden_dim, # 将LSTM的输出作为线性层的输入
                            out_features=1 # 由于情感分析只需要输出0或1，所以输出的维度是1
                            ) 
        self.sigmoid = nn.Sigmoid() # 线性层输出后，还需要过一下sigmoid
        
        # 给最后的全连接层加一个Dropout
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x, hidden):
        """
        x: 本次的输入，其size为(batch_size, 200)，200为句子长度
        hidden: 上一时刻的Hidden State和Cell State。类型为tuple: (h, c), 
        其中h和c的size都为(n_layers, batch_size, hidden_dim), 即(2, 200, 512)
        """
        # 因为一次输入一组数据，所以第一个维度是batch的大小
        batch_size = x.size(0) 
        
        # 由于embedding只接受LongTensor类型，所以将x转换为LongTensor类型
        x = x.long() 
        
        # 对x进行编码，这里会将x的size由(batch_size, 200)转化为(batch_size, 200, embedding_dim)
        embeds = self.embedding(x)
        
        # 将编码后的向量和上一时刻的hidden_state传给LSTM，并获取本次的输出和隐状态（hidden_state, cell_state）
        # lstm_out的size为 (batch_size, 200, 512)，200是单词的数量，由于是一个单词一个单词送给LSTM的，所以会产生与单词数量相同的输出
        # hidden为tuple(hidden_state, cell_state)，它们俩的size都为(2, batch_size, 512), 2是由于lstm有两层。由于是所有单词都是共享隐状态的，所以并不会出现上面的那个200
        lstm_out, hidden = self.lstm(embeds, hidden) 
        
        # 接下来要过全连接层，所以size变为(batch_size * 200, hidden_dim)，
        # 之所以是batch_size * 200=40000，是因为每个单词的输出都要经过全连接层。
        # 换句话说，全连接层的batch_size为40000
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # 给全连接层加个Dropout
        out = self.dropout(lstm_out)
        
        # 将dropout后的数据送给全连接层
        # 全连接层输出的size为(40000, 1)
        out = self.fc(out)
        
        # 过一下sigmoid
        out = self.sigmoid(out)
        
        # 将最终的输出数据维度变为 (batch_size, 200)，即每个单词都对应一个输出
        out = out.view(batch_size, -1)
        
        # 只去最后一个单词的输出
        # 所以out的size会变为(200, 1)
        out = out[:,-1]
        
        # 将输出和本次的(h, c)返回
        return out, hidden 
    
    def init_hidden(self, batch_size):
        """
        初始化隐状态：第一次送给LSTM时，没有隐状态，所以要初始化一个
        这里的初始化策略是全部赋0。
        这里之所以是tuple，是因为LSTM需要接受两个隐状态hidden state和cell state
        """
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
                 )
        return hidden
    
model = SentimentNet(len(words))
model.to(device)


criterion = nn.BCELoss()

lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2 # 一共训练两轮
counter = 0 # 用于记录训练次数
print_every = 1000 # 每1000次打印一下当前状态

for i in range(epochs):
    h = model.init_hidden(batch_size) # 初始化第一个Hidden_state
    
    for inputs, labels in train_loader: # 从train_loader中获取一组inputs和labels
        counter += 1 # 训练次数+1
        
        # 将上次输出的hidden_state转为tuple格式
        # 因为有两次，所以len(h)==2
        h = tuple([e.data for e in h]) 
        
        # 将数据迁移到GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清空模型梯度
        model.zero_grad()
        
        # 将本轮的输入和hidden_state送给模型，进行前向传播，
        # 然后获取本次的输出和新的hidden_state
        output, h = model(inputs, h)
        
        # 将预测值和真实值送给损失函数计算损失
        loss = criterion(output, labels.float())
        
        # 进行反向传播
        loss.backward()
        
        # 对模型进行裁剪，防止模型梯度爆炸
        # 详情请参考：javascript:void(0)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        
        # 更新权重
        optimizer.step()
        
        # 隔一定次数打印一下当前状态
        if counter%print_every == 0:
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()))

#-----------------------------評估模型性能
test_losses = [] # 记录测试数据集的损失
num_correct = 0 # 记录正确预测的数量
h = model.init_hidden(batch_size) # 初始化hidden_state和cell_state
model.eval() # 将模型调整为评估模式

# 开始评估模型
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze()) # 将模型四舍五入为0和1
    correct_tensor = pred.eq(labels.float().view_as(pred)) # 计算预测正确的数据
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    
print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))


def predict(sentence):
    # 将句子分词后，转换为数字
    sentences = [[word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]]
    
    # 将句子变为固定长度200
    sentences = pad_input(sentences, 200)
    
    # 将数据移到GPU中
    sentences = torch.Tensor(sentences).long().to(device)
    
    # 初始化隐状态
    h = (torch.Tensor(2, 1, 512).zero_().to(device),
         torch.Tensor(2, 1, 512).zero_().to(device))
    h = tuple([each.data for each in h])
    
    # 预测
    if model(sentences, h)[0] >= 0.5:
        print("positive")
    else:
        print("negative")

predict("The film is so boring")
predict("The actor is too ugly.")
