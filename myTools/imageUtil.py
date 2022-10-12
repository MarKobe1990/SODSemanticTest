import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.utils as utils
import torchvision.transforms as transforms
from PIL import ImageOps
import os.path as osp
import time
from math import *


# tensor chanel BGR
def merge_image_tensor_BGR(img_np_list, file_name, save_dir, n_row, tag=None):
    grid = utils.make_grid(torch.cat([img_RGB, img_RGB, img_RGB, img_RGB], dim=0))
    print(grid.size())
    input_tensor = grid.clone().detach().to(torch.device('cpu'))  # 到cpu
    utils.save_image(input_tensor, file_name)


def merge_image_np(img_np_list, file_name, save_dir, n_row, tag=None):
    img_tensor_list = []
    for img_np in img_np_list:
        img_tensor = transforms.ToTensor()(img_np).unsqueeze(dim=0)
        img_tensor_list.append(img_tensor)
    image_grid = utils.make_grid(torch.cat(img_tensor_list, dim=0), n_row).clone().detach().to(torch.device('cpu'))
    utils.save_image(image_grid, save_dir + file_name)


def merge_image_file_name(file_name_list, merge_img_name, save_dir, n_row, tag_dic=None):
    img_tensor_list = []
    shape = cv2.imread(file_name_list[0]).shape
    for file_name in file_name_list:
        img = cv2.imread(file_name)
        if img.shape != shape:
            img = cv2.resize(img, (shape[1], shape[0]))
        img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if file_name == file_name_list[-1]:
            if tag_dic is not None:
                i = 0
                for key, value in tag_dic.items():
                    text_img = '{0:<8} : {1:>6.4f}'.format(key, value)
                    # !图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                    cv2.putText(img_np, text_img, (10, 25 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    i = i + 1
        img_tensor = transforms.ToTensor()(img_np).unsqueeze(dim=0)
        img_tensor_list.append(img_tensor)
    image_grid = utils.make_grid(torch.cat(img_tensor_list, dim=0), n_row, pad_value=255).clone().detach().to(
    torch.device('cpu'))
    utils.save_image(image_grid, save_dir + merge_img_name)



def look_img(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


def inverse_color(image):
    height, width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i, j] = (255 - image[i, j])
    return img2


if __name__ == "__main__":
    # test = torch.rand((10, 1, 320, 320))
    # test = test[:, 0, :, :].unsqueeze(dim=1)
    # print(test.shape)
    # image = cv2.imread("data/NJU2K/000001_left.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_RGB = transforms.ToTensor()(image).unsqueeze(dim=0)
    # print(img_RGB.size())
    # grid = utils.make_grid(torch.cat([img_RGB, img_RGB, img_RGB, img_RGB], dim=0))
    # print(grid.size())
    # input_tensor = grid.clone().detach().to(torch.device('cpu'))  # 到cpu
    # utils.save_image(input_tensor, "out_cv.jpg")
    # cv2.imwrite("merge_image.jpg", image)

    # look_img(image)
    # image_list = [image, image, image, image]  # 这里存放你的图片帧列表
    # big_image = merge_image(image_list, 2, 2)  # x_num为行数、y_num为列数，x与y的积为图像列表的长度
    # cv2.imwrite("merge_image.jpg", big_image)
    src = './data/COME15K/test/COME-E/depths/COME_Easy_1007.png'
    depth = cv2.imread(src, 0)
    # depth_rev = inverse_color(depth)
    # eq_img = cv2.equalizeHist(depth)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    # cl1 = clahe.apply(depth_rev)
    # depth_rev_enhance = cv2.convertScaleAbs(depth_rev, 6050, 1)

    depth_resize=cv2.resize(depth, (360, 640))
    cv2.imshow('depth', depth)
    cv2.imshow('depth_resize', depth_resize)
    # cv2.imshow('depth_rev', depth_rev)
    # cv2.imshow('eq_img', eq_img)
    # cv2.imshow('cl1', cl1)
    key = cv2.waitKey(0)
