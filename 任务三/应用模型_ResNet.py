# coding:utf-8

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

#定义将cuda.tensor转化为numpy的函数
def tensor_to_np(x):
    temp = x.cpu().int()
    return temp.detach().numpy()

#定义混淆矩阵绘制函数
def confusion_matrix_drawing(pred, label, n=10): #pred,label均为经过转换后的list,n为分类个数
    confusion = confusion_matrix(label, pred)
    precision = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
    print('precision:%s' % precision)
    recall = confusion[1, 1] / (confusion[0, 1] + confusion[1, 1])
    print('recall:%s' % recall)
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

#定义transform
transform = transforms.Compose(
                    [transforms.Resize([224, 224]),
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

#加载数据
data = []
for i in range(0, 1000):
    file_name = r'./test/%s.png' % i
    temp = Image.open(file_name)
    data.append(transform(temp))

#加载原来的网络结构
ResNet = torchvision.models.resnet18()
ResNet.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.Linear(128, 32),
        nn.Linear(32, 2)
    )

#加载模型
net = torch.load(r'./ResNet_PCB.pt', map_location= torch.device('cpu'))
net.eval()

#加载答案
answer = open(r'E:\资料\人工智能\作业三\answer.csv')
answer = answer.read()
y_true = [int(i) for i in answer.split('\n')[1: -1]] #混淆矩阵的正确值

if __name__ == '__main__':
    y_predict = []
    correct = 0
    for pic in data:
        outputs_test = net(pic.unsqueeze(0)) #将三维的pic转化为四维，即batch为1，以对应bn层的输入
        _, predict = torch.max(outputs_test, 1) #用predict接受torch.max算出来的第二个张量
        y_predict += [i for i in tensor_to_np(predict)]  # 混淆矩阵的预测值

    for i in range(0, 1000):
        correct += (y_true[i] == y_predict[i])

    print('correct:', correct)
    print('final result:', correct / 1000)
    # 绘制混淆矩阵
    confusion_matrix_drawing(y_true, y_predict, 2)

#ResNet_PCB
#correct:885
#final result:0.885


