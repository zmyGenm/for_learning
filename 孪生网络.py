# coding:utf-8

import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
from sklearn.metrics import confusion_matrix

#定义一些超参
train_batch_size = 64        #训练时batch_size
train_number_epochs = 100     #训练的epoch

def imshow(img,text=None,should_save=False):
    #展示一幅tensor图像，输入是(C,H,W)
    npimg = img.numpy() #将tensor转为ndarray
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))) #转换为(H,W,C)
    plt.show()

def show_plot(iteration,loss):
    #绘制损失变化图
    plt.plot(iteration,loss)
    plt.show()

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

# 自定义Dataset类，__getitem__(self,index)每次返回(img1, img2, 0/1)
class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # 37个类别中任选一个
        should_get_same_class = random.randint(0, 1)  # 保证同类样本约占一半
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


#定义transfrom
transform = transforms.Compose([transforms.Resize((100, 100)),  # 有坑，传入int和tuple有区别
                                transforms.ToTensor()])

#加载数据集并划分为训练集、测试集、验证集
train_set = torchvision.datasets.ImageFolder(root=r'./data/train/')
test_set = torchvision.datasets.ImageFolder(root=r'./data/test')

# 定义图像dataset
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=train_set,
                                        transform=transform,
                                        should_invert=False)

# 定义图像dataloader
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              batch_size=train_batch_size)

vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        batch_size=8)
example_batch = next(iter(vis_dataloader)) #生成一批图像
#其中example_batch[0] 维度为torch.Size([8, 1, 100, 100])
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated, nrow=8))
print(example_batch[2].numpy())

#定义测试的dataset和dataloader
#定义图像dataset
transform_test = transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor()])
siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=test_set,
                                        transform=transform_test,
                                        should_invert=False)

#定义图像dataloader
test_dataloader = DataLoader(siamese_dataset_test,
                            shuffle=True,
                            batch_size=1)

# 搭建模型
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            )
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    # 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

net = SiameseNetwork().cuda()  # 定义模型且移至GPU
criterion = ContrastiveLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 定义优化器

counter = []
loss_history = []
iteration_number = 0

#可视化网络结构
from torchviz import make_dot
x1 = torch.randn(1, 1, 100, 100).requires_grad_(True)  # 定义一个网络的输入值
x1 = x1.cuda()
x2 = torch.randn(1, 1, 100, 100).requires_grad_(True)  # 定义一个网络的输入值
x2 = x2.cuda()
y = net(x1, x2)    # 获取网络的预测值

MyConvNetVis = make_dot(y, params=dict(list(net.named_parameters()) + [('x1', x1), ('x2', x2)]))
MyConvNetVis.format = "png"
# 指定文件生成的文件夹
MyConvNetVis.directory = "网络结构"
# 生成文件
MyConvNetVis.view()

# 开始训练
for epoch in range(0, train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        # img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()  # 数据移至GPU
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))

show_plot(counter, loss_history)

#生成对比图像
dataiter = iter(test_dataloader)
x0,_,_ = next(dataiter)

for i in range(10):
    _,x1,label2 = next(dataiter)
    concatenated = torch.cat((x0,x1),0)
    output1,output2 = net(x0.cuda(),x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

#输出test集结果
#定义transform
transform = transforms.Compose(
                        [transforms.Resize([100, 100]),
                        transforms.ToTensor()])

# 加载数据
data = []
for i in range(0, 1000):
    file_name = r'./test/%s.png' % i
    temp = Image.open(file_name).convert("L")
    data.append(transform(temp))

#加载答案
answer = open(r'E:\资料\人工智能\作业三\answer.csv')
answer = answer.read()
y_true = [int(i) for i in answer.split('\n')[1: -1]]  # 混淆矩阵的正确值

#加载用于对比的abnormal样本和normal样本
pic_abnormal = transform(Image.open(r'train/abnormal/0.png').convert("L"))
pic_abnormal = pic_abnormal.unsqueeze(0) #将三维的pic_abnormal转化为四维，即batch为1，以对应bn层的输入
pic_normal = transform(Image.open(r'train/normal/0.png').convert("L"))
pic_normal = pic_normal.unsqueeze(0) #将三维的pic_normal转化为四维，即batch为1，以对应bn层的输入

#将预测结果导出
y_pred = []
for pic in data:
    pic = pic.unsqueeze(0) #将三维的pic转化为四维，即batch为1，以对应bn层的输入
    output1, output2 = net(pic_abnormal.cuda(), pic.cuda())
    euclidean_distance1 = F.pairwise_distance(output1, output2)
    output3, output4 = net(pic_normal.cuda(), pic.cuda())
    euclidean_distance2 = F.pairwise_distance(output3, output4)
    euclidean_distance = torch.cat((euclidean_distance1, euclidean_distance2), 0)
    __, result = torch.min(torch.softmax(euclidean_distance, 0), 0) #计算出距离小的才是分类
    y_pred.append(result.item())

#对比标签和模型结果
correct = 0
for i in range(0, 1000):
    correct += (y_true[i] == y_pred[i])

print('correct:', correct)
print('final result:', correct / 1000)
# 绘制混淆矩阵
confusion_matrix_drawing(y_true, y_pred, 2)

#correct: 838
#final result: 0.838
