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


def show_batch(imgs):
   grid = utils.make_grid(imgs,nrow=5)
   plt.imshow(grid.numpy().transpose((1, 2, 0)))
   plt.title('Batch from dataloader')


def pytorch1():
    a = 1
    img_data  = torchvision.datasets.ImageFolder( 'D:/Thesis/test',
                                                transform=transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor()])
                                                )
    
    print(len(img_data))
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=20,shuffle=True)
    print(len(data_loader))    
    
    for i, (batch_x, batch_y) in enumerate(data_loader):
        if(i<4):
            print(i, batch_x.size(), batch_y.size())
    
            show_batch(batch_x)
            plt.axis('off')
            plt.show()
            
def trya():
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
    
    # 画图
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.show()


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,input):
        ourput = input + 1
        return ourput
          

tudui = Tudui()
x = torch.tensor(1.0)
ourput = tudui(x)
print(ourput)

        
if __name__ =="__main__": trya()