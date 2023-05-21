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
train_loader = DataLoader(datasets.CIFAR10('data', train=True, download=False, transform=transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(), #使用数据增强
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])),
                          batch_size=100, shuffle=True, num_workers=2)

#加载测试集
test_loader = DataLoader(datasets.CIFAR10('data', train=False, download=False, transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])),
                          batch_size=100, shuffle=False, num_workers=2)

#定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5), #相比于之前的简单CNN，将卷积核个数大大增加
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, 3), #相比于之前的简单CNN，将卷积核个数大大增加
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), #相比于之前的简单CNN，将卷积核个数大大增加
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            nn.Linear(128 * 4 * 4, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

#定义设备，设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001

#实例化模型，定义损失函数与优化算法
net = Net().to(device)
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
    net.eval()
    correct = 0; total = 0
    y_predict = []
    y_true = []
    for data_test in test_loader:
        inputs, labels = data_test
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        outputs_test = net(inputs)
        _, predict = torch.max(outputs_test, 1) #用predict接受torch.max算出来的第二个张量
        total += labels.size(0)
        correct += (predict == labels).sum()
        y_true += [i for i in tensor_to_np(labels)] #混淆矩阵的预测值
        y_predict += [i for i in tensor_to_np(predict)] #混淆矩阵的真实值
    print('correct:', correct)
    print('final result:', correct / total)
    #绘制混淆矩阵
    confusion_matrix_drawing(y_true, y_predict)

#best:acc epoch=20
#correct: tensor(7573, device='cuda:0')
#final result: tensor(0.7573, device='cuda:0')
