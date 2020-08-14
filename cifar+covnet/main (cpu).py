import torch
from torchvision import datasets, transforms
from  torch.utils.data import  DataLoader
from  torch import nn
from torch import  optim
import torch.nn.functional as F
from  CovNet import CovNet


def main():
    batch_size=218
    cifar_train=datasets.CIFAR10("cifar",train=True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)
    cifar_train=DataLoader(cifar_train,batch_size=batch_size,shuffle=True)
    cifar_test=datasets.CIFAR10("cifar",train=False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]))
    cifar_test=DataLoader(cifar_test,batch_size=batch_size,shuffle=True)


    model=CovNet()
    # use Cross Entropy loss
    # The cross entropy function of pytorch includes a softmax step
    criteon = nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(10000):
        model.train()
        for batch_idx ,(data,label) in enumerate(cifar_train):
            #data:[batch_size,3,32,32]
            #label:[batch_size,]
            #logits:[batch_size,10]
            #loss:tensor scalar

            #forward
            logits=model(data)
            loss=criteon(logits,label)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch is: %d ,loss is: %f"%(epoch,loss.item()))

        #model test
        #model test is without no autograd
        model.eval()
        with torch.no_grad():
            total_num = 0
            total_correct = 0
            for data, label in cifar_test:
                # data:[batch_size,3,32,32]
                # label:[batch_size,]

                # logits:[batch_size,10]
                logits = model(data)
                # predict:[batch_size,1]
                predict = logits.argmax(dim=1)
                # [batch_size,1] vs [batch_size,1] →scalar tensor
                total_correct += torch.eq(predict, label).float().sum().item()
                total_num += data.size(0)
            acc = total_correct / total_num
            print("epoch is: %d ,acc is: %f" % (epoch, acc))

if __name__=="__main__":
    main()