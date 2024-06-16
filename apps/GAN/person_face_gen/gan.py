import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import torchvision.utils as vutils

from train import train
from loss_func import loss_func
from utils import get_discriminator,get_generator
import data_load
from device import device
from plot import plot_original_dataset,loss_plot,image_show
from const import ngpu, nz, ngf, nc, ndf, batch_size, workers, image_size
from generator import Generator 
from discriminator import Discriminator

if __name__ == '__main__':
    #加载数据
    dataset=data_load.dateset_load()
    # 创建生成器和判别器
    netG=get_generator(nz, ngf, nc)
    netD=get_discriminator(nc, ndf)
    #损失函数相关
    criterion, fixed_noise, real_label, fake_label, optimizerD, optimizerG = loss_func(netD, netG, device)
    #训练
    G_losses,D_losses,img_list=train(dataset,netD,netG,fixed_noise,criterion,real_label,fake_label,optimizerD,optimizerG,device)
    #绘制loss图
    loss_plot(G_losses,D_losses)
    #绘制生成图像
    image_show(img_list)
    #保存模型
    Generator.save_model(netG,'./models')
    Discriminator.save(netD,'./models')
    
    #保存图片
    os.makedirs('./images', exist_ok=True)
    vutils.save_image(img_list[-1], './images/fake_images.png', normalize=True)

def test():
    #绘制原始数据图像
    # plot_original_dataset(dataset,device.device)
    #测试生成器
    # test_generator(nz, ngf, nc)
    #测试判别器
    # test_discriminator(nc, ndf)
    return