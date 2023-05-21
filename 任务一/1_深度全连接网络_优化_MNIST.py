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
train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.1037,), (0.3081,))])),
                          batch_size=90, shuffle=True)

#加载测试集
test_loader = DataLoader(datasets.MNIST('data', train=False, download=True, transform=transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Normalize((0.1037,), (0.3081,))])),
                          batch_size=64, shuffle=True)

#定义网络
class deepnet(nn.Module):
    def __init__(self):
        super(deepnet, self).__init__()
        self.first_layer = nn.Linear(784, 360)
        self.second_layer = nn.Linear(360, 120)
        self.third_layer = nn.Linear(120, 60)
        self.last_layer = nn.Linear(60, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.first_layer(x))
        x = F.dropout(self.second_layer(x))
        x = F.relu(self.third_layer(x))
        x = self.last_layer(x)
        return x

#添加设备，设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001

#实例化模型，设置损失函数与优化算法
net = deepnet().to(device)
criterion = nn.CrossEntropyLoss() #损失函数：交叉熵
optimizer = optim.Adam( # 优化算法
    net.parameters(),
    lr=LR,
)

#可视化网络结构
from torchviz import make_dot
x = torch.randn(1, 1, 28, 28).requires_grad_(True)  # 定义一个网络的输入值
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
        for i,data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad() #梯度归零
            outputs = net(inputs)
            loss = criterion(outputs, labels) #得到损失函数
            loss.backward() #反向传播
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
        _, predict = torch.max(outputs_test, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
        y_true += [i for i in tensor_to_np(labels)] #混淆矩阵的预测值
        y_predict += [i for i in tensor_to_np(predict)] #混淆矩阵的真实值
    print('correct:', correct)
    print('final result:', correct/total)

    #绘制混淆矩阵
    confusion_matrix_drawing(y_true, y_predict)

#best:acc epoch=20
#correct: tensor(9773, device='cuda:0')
#final result: tensor(0.9773, device='cuda:0')
