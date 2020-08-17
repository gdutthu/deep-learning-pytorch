import torch
from torchvision import datasets, transforms
from  torch import nn
from torch import  optim
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10)
        )

    def forward(self,x):
        x=self.model(x)
        return  x


def load_data(batch_size=200):
    print("Loading Data......")
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)
    return  train_loader,test_loader


def model_train(train_loader,test_loader):
    learning_rate = 0.001
    num_epoch = 200

    net = MLP()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    criteon = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        for batch_idx,(data,target) in enumerate(train_loader):
            data=data.view(-1,28*28)
            logit=net(data)
            loss=criteon(logit,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            logits = net(data)
            test_loss += criteon(logits, target).item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).float().sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

if __name__=="__main__":
    train_loader, test_loader = load_data(batch_size=200)
    model_train(train_loader, test_loader)
