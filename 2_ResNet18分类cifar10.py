# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#定义将cuda.tensor转化为numpy的函数
def tensor_to_np(x):
    temp = x.cpu().int()
    return temp.detach().numpy()

#定义混淆矩阵绘制函数
def confusion_matrix_drawing(pred, label, n=10): #pred,label均为经过转换后的list,n为问题分类个数
    confusion = confusion_matrix(label, pred)
    _, ax = plt.subplots() #开始绘制
    plt.imshow(confusion, cmap=plt.cm.Blues) #绘制热力图，颜色为蓝
    plt.title('confusion_matrix')
    plt.xlabel('true')
    plt.ylabel('predict')
    plt.xticks([i for i in range(n)]) #该问题为10分类问题
    plt.yticks([i for i in range(n)])
    #填充数字
    for i in range(n):
        for j in range(n):
            ax.text(j, i, confusion[i, j],
                    ha="center", va="center", color="black")
    plt.colorbar()
    plt.show()

#定义将cuda.tensor转化为numpy的函数
def tensor_to_np(x):
    temp = x.cpu().int()
    return temp.detach().numpy()

#定义混淆矩阵绘制函数
def confusion_matrix_drawing(pred, label, n=10): #pred,label均为经过转换后的list,n为分类个数
    confusion = confusion_matrix(label, pred)
    _, ax = plt.subplots() #开始绘制
    plt.imshow(confusion, cmap=plt.cm.Blues) #绘制热力图，颜色为蓝
    plt.title('confusion_matrix')
    plt.xlabel('true')
    plt.ylabel('predict')
    plt.xticks([i for i in range(n)]) #该问题为10分类问题
    plt.yticks([i for i in range(n)])
    #填充数字
    for i in range(n):
        for j in range(n):
            ax.text(j, i, confusion[i, j],
                    ha="center", va="center", color="black")
    plt.colorbar()
    plt.show()

#加载训练集
train_loader = DataLoader(datasets.CIFAR10('data', train=True, download=False, transform=transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(), #使用数据增强
                     transforms.Resize(224),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                          batch_size=64, shuffle=True, num_workers=2)

#加载测试集
test_loader = DataLoader(datasets.CIFAR10('data', train=False, download=False, transform=transforms.Compose(
                    [transforms.Resize(224),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                          batch_size=64, shuffle=False, num_workers=2)

#定义网络
#单个包含残差结构的网络
class conv_n(nn.Module):
    def __init__(self, inchannel, lastchannel, shortcut=False, stride=1): #inchannel表示每个卷积层的通道数，lastchannel表示上一个包含残差结构的网络的输出通道数,shortcut表示有无虚线残差结构
        super(conv_n, self).__init__()
        self.conv1 = nn.Conv2d(lastchannel, inchannel, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inchannel, inchannel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(inchannel)
        self.relu2 = nn.ReLU()
        if shortcut:
            self.short = nn.Conv2d(lastchannel, inchannel, 1, stride=2)
        else:
            self.short = False

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if self.short:
            return out + self.short(x)
        else:
            return out + x


#ResNet18
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
                shortcut = True
            else:
                stride = 1
                shortcut = False
            self.conv2to5.add_module('conv%d' % (k+2), nn.Sequential())
            inchannel = 64 * (2 ** k)
            # 每一个单独的含有残差结构的网络内部
            for j in range(num):
                if j == 0:
                    self.conv2to5[k].add_module('conv%d_%d' % ((k + 2), j),
                                                conv_n(inchannel, lastchannel, shortcut, stride))
                else:
                    self.conv2to5[k].add_module('conv%d_%d' % ((k + 2), j), conv_n(inchannel, lastchannel))
                lastchannel = inchannel
            else:
                self.conv2to5[k].add_module('conv%d_ReLU' % (k + 2), nn.ReLU())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Sequential(
            nn.Linear(512, 10)
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
net = ResNet([2, 2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss() #损失函数：交叉熵
optimizer = optim.Adam( #优化算法
    net.parameters(),
    lr=LR
)

#可视化网络结构
from torchviz import make_dot
x = torch.randn(1, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
x = x.cuda()
y = net(x)    # 获取网络的预测值

MyConvNetVis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
MyConvNetVis.format = "png"
# 指定文件生成的文件夹
MyConvNetVis.directory = "pictures"
# 生成文件
MyConvNetVis.view()

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
        y_predict = []
        y_true = []
        for data_test in test_loader:
            inputs, labels = data_test
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            outputs_test = net(inputs)
            _, predict = torch.max(outputs_test, 1) #用predict接受torch.max算出来的第二个张量
            total += labels.size(0)
            correct += (predict == labels).sum()
            y_true += [i for i in tensor_to_np(labels)]  # 混淆矩阵的预测值
            y_predict += [i for i in tensor_to_np(predict)]  # 混淆矩阵的真实值
        print('correct:', correct)
        print('final result:', correct / total)
        # 绘制混淆矩阵
        confusion_matrix_drawing(y_true, y_predict)
        #保存模型
        torch.save(net, 'resnet18_model_epoch20.pt')

#best_acc：epoch=20
#correct: tensor(9013, device='cuda:0')
#final result: tensor(0.9013, device='cuda:0')
