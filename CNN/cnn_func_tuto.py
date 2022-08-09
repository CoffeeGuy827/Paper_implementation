import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=5,stride=1)
#         self.conv2 = nn.Conv2d(in_channels=3, out_channels=10,kernel_size=5,stride=1)
#         self.fc1 =nn.Linear(10*12*12,50)
#         self.fc2=nn.Linear(50,10)
#
#     def forward(self,x):
#         print("연산 전", x.size())
#         x = F.relu(self.conv1(x))
#         print("conv1 연산 후", x.size())
#         x = F.relu(self.conv2(x))
#         print("conv2 연산 후", x.size())
#         x = x.view(-1,10*12*12)
#         print("차원 감소 후", x.size())
#         x = F.relu(self.fc1(x))
#         print("fc1 연산 후", x.size())
#         x = self.fc2(x)
#         print("fc2 연산 후", x.size())
#         return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(10 * 5 * 5, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self,x):
        print("연산 전", x.size())
        x = F.relu(self.max_pool1(x))
        print("max_pool1 연산 후", x.size())
        x = F.relu(self.max_pool2(x))
        print("max_pool2 연산 후", x.size())
        x = x.view(-1, 10 * 5 * 5)
        print("차원 감소 후", x.size())
        x = F.relu(self.fc1(x))
        print("fc1 연산 후", x.size())
        x = self.fc2(x)
        print("fc2 연산 후", x.size())
        return x
cnn = CNN()
output = cnn(torch.randn(10,1,20,20))