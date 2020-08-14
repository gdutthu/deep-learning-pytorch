import torch
from torchvision import datasets, transforms
import  numpy as np


#one-hot函数处理
def one_hot(label):
    class_num=len(np.unique(label))
    sample_num=len(label)
    one_hot_label = torch.zeros(sample_num, class_num).scatter_(1, label, 1)
    return  one_hot_label





print("Loading Data......")
"""加载MNIST数据集，本地数据不存在会自动下载"""
"""在加载train_data时，设置download=True进行下载mnist数据集，所以在加载test_data时不需要再进行下载"""
train_data = datasets.MNIST(root='./data/',train=True,
                            transform=transforms.ToTensor(),download=True)
test_data = datasets.MNIST(root='./data/',train=False,
                           transform=transforms.ToTensor())
batch_size=128
# 返回一个数据迭代器
# shuffle：是否打乱顺序
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,
                                          shuffle=False)
# return train_loader, test_loader
# print(train_data.shape)
