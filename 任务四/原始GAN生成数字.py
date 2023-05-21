#coding:utf-8

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images_mnist_gan", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# 生成原始噪点数据大小--latent_dim
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
# print(img_shape) 1 ,28,28
# print(int(np.prod(img_shape))) 784
cuda = True if torch.cuda.is_available() else False


# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 参数 进入32 出来 64  归一化
        def block(in_feat, out_feat, normalize=True):
            # 对传入数据应用线性转换（输入节点数，输出节点数）
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # 批规范化
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
                # 激活函数
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 模型定义
        self.model = nn.Sequential(

            *block(opt.latent_dim, 128, normalize=False),

            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            # np.prod 用来计算所有元素的乘积
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    # 正向传播
    def forward(self, z):
        img = self.model(z)  # shape 64 784
        img = img.view(img.size(0), *img_shape)  # 64 1 28 28
        return img


# 判别模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 64 1 28 28 =>64 784
        validity = self.model(img_flat)  # 64 784 =>64 1

        return validity


# Loss function 类似 目标值-得到值 的差值一种运算
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# 如果有gpu
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("data/mnist", exist_ok=True)
print(opt.img_size)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            # 其他地方也许是Resize((opt.img_size,opt.img_size)) 也就是（(28,28)）因为后续重塑格式类似于（64，1，28，28）
            # 这里是（28）  后面重塑格式类似于（64，1，28*28）
            # transforms.Normalize([0.5], [0.5])  这是单通道数据集
            # transforms.Normalize((0.5,0.5,0.5), (0.5),(0.5),(0.5))  三通道数据集
            # 图片三个通道
            # 前一个(0.5,0.5,0.5)是设置的mean值 后一个(0.5,0.5,0.5)是是设置各通道的标准差
            # 其作用就是先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    # 一次多少个处理，小图片一般64个
    batch_size=opt.batch_size,
    # 数据集打乱，洗牌
    shuffle=True,
)

# Optimizers 优化器
# lr=opt.lr学习率
# betas (Tuple[float, float]，可选):用于计算的系数
# 梯度及其平方的运行平均值(默认值:(0.9,0.999))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 判断是否有gpu
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

#可视化网络结构
from torchviz import make_dot
x_g = Variable(Tensor(np.random.normal(0, 1, (64, opt.latent_dim))))
x_d = torch.randn(1, 1, 28, 28).requires_grad_(True)  # 定义一个网络的输入值
x_d = x_d.cuda()
y_g = generator(x_g)    # 获取网络的预测值
y_d = discriminator(x_d)

MyConvNetVis1 = make_dot(y_g, params=dict(list(generator.named_parameters()) + [('x', x_g)]))
MyConvNetVis1.format = "png"
MyConvNetVis2 = make_dot(y_d, params=dict(list(discriminator.named_parameters()) + [('x', x_d)]))
MyConvNetVis2.format = "png"
# 指定文件生成的文件夹
MyConvNetVis1.directory = "picture_g"
MyConvNetVis2.directory = "picture_d"
# 生成文件
MyConvNetVis1.view()
MyConvNetVis2.view()

opt.n_epochs = 50
for epoch in range(opt.n_epochs):
    # dataloader中的数据是一张图片对应一个标签，所以imgs对应的是图片，_对应的是标签，而i是enumerate输出的功能
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        # 这部分定义的相当于是一个标准，vaild可以想象成是64行1列的向量，就是为了在后面计算损失时，和1比较；fake也是一样是全为0的向量，用法和1的用法相同。
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        # 这句是将真实的图片转化为神经网络可以处理的变量。变为Tensor
        # print(type(imgs)) Tensor
        real_imgs = Variable(imgs.type(Tensor))
        # print(type(real_imgs)) Tensor
        # -----------------
        #  Train Generator
        # -----------------

        # optimizer.zero_grad()意思是把梯度置零
        # 每次的训练之前都将上一次的梯度置为零，以避免上一次的梯度的干扰
        optimizer_G.zero_grad()

        # Sample noise as generator input
        # 这部分就是在上面训练生成网络的z的输入值，np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)的意思就是
        # 64个噪音（基础值为100大小的） 0，代表正态分布的均值，1，代表正态分布的方差
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # Generate a batch of images 返回一个批次即64个
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        # 计算这64个图片总损失  生成器损失
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # 反向传播
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # 梯度清零
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # 判别器判别真实图片是真的的损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        # 判别器判别假的图片是假的的损失
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        # 判别器去判别真实图片是真的的概率大，并且判别假图片是真的的概率小，说明判别器越准确所以说是maxD，
        # 生成器就是想生成真实的图片来迷惑判别器，所以理论上想让生成器生成真实的图片概率大，
        # 由于公式第二部分表示生成器的损失，G（z）前有个负号，所以如果结果小则证明G生成的越真实，所以说minG
        d_loss = (real_loss + fake_loss) / 2

        # 反向传播
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images_mnist_gan/%d.png" % batches_done, nrow=5, normalize=True)
