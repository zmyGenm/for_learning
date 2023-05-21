# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#加载训练集
train_loader = DataLoader(datasets.CIFAR10('data', train=True, download=False, transform=transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(), #使用数据增强
                     transforms.Resize(224),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                          batch_size=16, shuffle=True, num_workers=2)

#加载测试集
test_loader = DataLoader(datasets.CIFAR10('data', train=False, download=False, transform=transforms.Compose(
                    [transforms.Resize(224),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                          batch_size=16, shuffle=False, num_workers=2)

#定义网络
#单个包含残差结构的网络
class conv_n(nn.Module):
    def __init__(self, inchannels, lastchannel, shortcut_simple=False, stride=1): #inchannels为输入的通道数，lastchannel为上一个网络的输出通道数，shortcut_simple为是否有“虚线”残差结构
        super(conv_n, self).__init__()
        self.channels = [inchannels, inchannels, inchannels*4] #channels为内部各卷积层通道数组成的列表
        self.conv_n = nn.Sequential(
            nn.Conv2d(lastchannel, self.channels[0], 1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[1], 3, stride=stride, padding=1),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.channels[1], self.channels[2], 1),
            nn.BatchNorm2d(self.channels[2])
        )
        if shortcut_simple: #残差结构复杂
            self.shortcut = nn.Conv2d(lastchannel, inchannels*4, 1, stride=stride)
            self.temp = None
        else: #残差结构简单
            self.shortcut = False
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_n(x)
        if self.shortcut:
            out += self.shortcut(x)
        else: #残差结构简单
            out += x
        return self.relu(out)

#ResNet50
class ResNet(nn.Module):
    def __init__(self, block_num): #block_num为记录每种包含残差结构网络的数量的列表
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2to5 = nn.Sequential()
        lastchannel = 64
        #通过循环产生有相似结构的网络
        for k, num in enumerate(block_num):
            if k != 0:
                stride = 2
            else:
                stride = 1
            self.conv2to5.add_module('conv%d' % (k+2), nn.Sequential())
            inchannels = 64 * (2 ** k)
            #每一个单独的含有残差结构的网络内部
            for j in range(num):
                if j == 0:
                    self.conv2to5[k].add_module('conv%d_%d' % ((k+2), j), conv_n(inchannels, lastchannel, True, stride))
                else:
                    self.conv2to5[k].add_module('conv%d_%d' % ((k+2), j), conv_n(inchannels, lastchannel))
                lastchannel = 4 * inchannels
            else:
                self.conv2to5[k].add_module('conv%d_ReLU' % (k+2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Sequential(
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv2to5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

#定义设备，设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001

#实例化模型，定义损失函数与优化算法
net = ResNet([3, 4, 6, 3]).to(device)
criterion = nn.CrossEntropyLoss() #损失函数：交叉熵
optimizer = optim.Adam( #优化算法
    net.parameters(),
    lr=LR
)

#开始训练
epoch = 20
if __name__ == '__main__':
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad() #梯度归零

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            outputs = net(inputs)
            loss = criterion(outputs, labels) #计算损失
            loss.backward() #损失函数反向传播
            optimizer.step() #更新梯度

            #print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

    #开始测试
    with torch.no_grad():
        net.eval()
        correct = 0; total = 0
        for data_test in test_loader:
            inputs, labels = data_test
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            outputs_test = net(inputs)
            _, predict = torch.max(outputs_test, 1) #用predict接受torch.max算出来的第二个张量
            total += labels.size(0)
            correct += (predict == labels).sum()
        print('correct:', correct)
        print('final result:', correct / total)
        #保存模型
        torch.save(net, './resnet_model_20.pt')

#best_acc：epoch=10
#correct: tensor(7923, device='cuda:0')
#final result: tensor(0.7923, device='cuda:0')
