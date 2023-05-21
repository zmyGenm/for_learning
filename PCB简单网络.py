# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
    plt.xticks([i for i in range(n)]) #该问题为2分类问题
    plt.yticks([i for i in range(n)])
    #填充数字
    for i in range(n):
        for j in range(n):
            ax.text(j, i, confusion[i, j],
                    ha="center", va="center", color="black")
    plt.colorbar()
    plt.show()

#定义transforms
transforms = transforms.Compose(
                    [transforms.RandomHorizontalFlip(), #使用数据增强
                     transforms.Resize([224, 224]),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

#加载数据集并划分为训练集、测试集、验证集
whole_dataset = datasets.ImageFolder(root=r'./train', transform=transforms)
whole_train_set, test_set = train_test_split(whole_dataset, test_size=0.2, random_state=42)
train_set, dev_set = train_test_split(whole_train_set, test_size=0.1, random_state=42)

#加载部分训练集和完整训练集
whole_train_loader = DataLoader(whole_train_set, batch_size=4, shuffle=True, num_workers=2)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

#加载测试集
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

#加载验证集
dev_loader = DataLoader(dev_set, batch_size=4, shuffle=False, num_workers=2)

#定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 5, 3, stride=2),
        nn.Conv2d(5, 3, 3, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(3)
        )
        self.fc = nn.Sequential(
        nn.Linear(3*9*9, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(x.size(0), -1))
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

#可视化网络结构
from torchviz import make_dot
x = torch.randn(1, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
x = x.cuda()
y = net(x)    # 获取网络的预测值

MyConvNetVis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
MyConvNetVis.format = "png"
# 指定文件生成的文件夹
MyConvNetVis.directory = "网络结构"
# 生成文件
MyConvNetVis.view()

#开始训练
epoch = 40
if __name__ == '__main__':
    loss_lst = []
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(whole_train_loader):
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
                loss_lst.append(sum_loss/100)
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0

    x_lst = [i for i in range(len(loss_lst))]
    plt.plot(x_lst, loss_lst)
    plt.show()

    #开始测试
    with torch.no_grad():
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
            y_true += [i for i in tensor_to_np(labels)]  # 混淆矩阵的真实值
            y_predict += [i for i in tensor_to_np(predict)]  # 混淆矩阵的预测值
        print('correct:', correct)
        print('final result:', correct / total)
        # 绘制混淆矩阵
        confusion_matrix_drawing(y_true, y_predict, 2)
        #保存模型
        #torch.save(net, './简单卷积PCB.pt')

#best_acc：epoch=40
#correct: tensor(97, device='cuda:0')
#final result: tensor(0.8083, device='cuda:0')
