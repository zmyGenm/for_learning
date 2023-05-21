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

#定义多个不同保留层数的网络
resnet_list = []
for save_layers in range(-5, 0):
    ResNet = torchvision.models.resnet18(pretrained=True)
    ResNet.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.Linear(128, 32),
        nn.Linear(32, 2)
    )
    save_layers = 0
    for para in list(ResNet.parameters())[:0-save_layers]:
        para.requires_grad = False
    resnet_list.append(ResNet)

#定义设备，设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001

#实例化模型，定义损失函数与优化算法
net_list = [ResNet.to(device) for ResNet in resnet_list]
optimizet_list = []
criterion = nn.CrossEntropyLoss() #损失函数：交叉熵
for net in net_list:
    optimizer = optim.Adam( #优化算法
        net.parameters(),
        lr=LR
    )
    optimizet_list.append(optimizer)

#开始训练
epoch = 1
if __name__ == '__main__':
    #挑选最优超参数
    best_list = []
    for index, net in enumerate(net_list):
        optimizer = optimizet_list[index]
        for epoch in range(epoch):
            sum_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                optimizer.zero_grad()  # 梯度归零
                outputs = net(inputs)
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 损失函数反向传播
                optimizer.step()  # 更新梯度
        print('网络%s训练完毕' % index)

       #转入验证集
        net.eval()
        correct = 0
        total = 0
        for data_test in dev_loader:
            images, labels = data_test
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs_test = net(images)
            _, predict = torch.max(outputs_test, 1)  # 只需要计算出来的第二个tensor，故在前面放一个_变量，使得predict为所要求的值
            total += labels.size(0)
            correct += (predict == labels).sum()
        best_list.append(correct / total)
    best = best_list.index(max(best_list))
    print('最佳网络为网络%s，准确率为%s' % (best, correct/total))

    # 开始训练
    epoch = 10
    net = net_list[best]
    optimizer = optimizet_list[best]
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
            outputs_test = net(inputs)
            _, predict = torch.max(outputs_test, 1) #用predict接受torch.max算出来的第二个张量
            total += labels.size(0)
            correct += (predict == labels).sum()
            y_true += [i for i in tensor_to_np(labels)]  # 混淆矩阵的预测值
            y_predict += [i for i in tensor_to_np(predict)]  # 混淆矩阵的真实值
        print('correct:', correct)
        print('final result:', correct / total)
        # 绘制混淆矩阵
        confusion_matrix_drawing(y_true, y_predict, 2)
        #保存模型
        #torch.save(net, './resnet18_model.pt')

#best_acc：epoch=
#correct: tensor(, device='cuda:0')
#final result: tensor(, device='cuda:0')
