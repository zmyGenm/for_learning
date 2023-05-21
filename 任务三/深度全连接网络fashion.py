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

#加载训练集
train_loader = DataLoader(datasets.FashionMNIST('data', train=True, download=False, transform=transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.1037,), (0.3081,))])),
                          batch_size=100, shuffle=True)

#加载测试集
test_loader = DataLoader(datasets.FashionMNIST('data', train=False, download=False, transform=transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.1037,), (0.3081,))])),
                          batch_size=64, shuffle=True)

#定义网络
class all_connect_net(nn.Module):
    def __init__(self):
        super(all_connect_net, self).__init__()
        self.first_layer = nn.Linear(784, 360)
        self.second_layer = nn.Linear(360, 120)
        self.third_layer = nn.Linear(120, 60)
        self.last_layer = nn.Linear(60, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.first_layer(x))
        x = torch.sigmoid(self.second_layer(x))
        x = torch.sigmoid(self.third_layer(x))
        x = self.last_layer(x)
        return x

#加载设备，设置初始参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#batch_size = 64
LR = 0.001

#实例化模型，设置损失函数与优化算法
net = all_connect_net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    net.parameters(),
    lr=LR,
)

#开始训练
epoch = 10
if __name__ == '__main__':
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad() #将梯度归零
            outputs = net(inputs)
            loss = criterion(outputs, labels) #得到损失函数
            loss.backward() #将损失反向传播
            optimizer.step() #梯度更新

            #print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    #开始测试
    net.eval()
    correct = 0
    total = 0
    y_predict = []
    y_true = []
    for data_test in test_loader:
        images, labels = data_test
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs_test = net(images)
        _, predict = torch.max(outputs_test, 1) #只需要计算出来的第二个tensor，故在前面放一个_变量，使得predict为所要求的值
        total += labels.size(0)
        correct += (predict == labels).sum()
        y_true += [i for i in tensor_to_np(labels)] #混淆矩阵的预测值
        y_predict += [i for i in tensor_to_np(predict)] #混淆矩阵的真实值
    print('correct:', correct)
    print('final result:', correct/total)

    #绘制混淆矩阵
    confusion_matrix_drawing(y_true, y_predict)
