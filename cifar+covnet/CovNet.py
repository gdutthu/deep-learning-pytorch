import torch
from  torch import nn
import torch.nn.functional as F

class CovNet(nn.Module):
    def __init__(self):
        super(CovNet,self).__init__()
        self.cov_unit=nn.Sequential(
            #x=[batchSize,3,32,32]→[batchSize,6,,]→[batchSize,16,5,5]
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        #flatten
        #fc unit
        self.fc_unit=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self,x):
        """
        :param x: [batchsize,3,32,32]
        :return:
        """
        batchsize=x.size(0)
        # [batchsize, 3, 32, 32]→[batchsize, 16, 5, 5]
        x=self.cov_unit(x)
        # [batchsize, 16, 5, 5]→[batchsize, 16*5*5]
        x=x.view(batchsize,16*5*5)
        # [batchsize, 16*5*5]→[batchsize, 10]
        logits=self.fc_unit(x)
        return  logits
