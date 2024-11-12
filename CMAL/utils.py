import os

import cv2
import numpy as np
import random
import torch
import torchvision
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F
from basic_conv import *
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.cluster import KMeans
from PIL import TiffImagePlugin
import logging
from sklearn.model_selection import StratifiedShuffleSplit
# import torchvision.transforms as transforms

# 注册.tif文件的打开扩展
TiffImagePlugin.READ_LIBTIFF = True
def show_image(inputs):
    inputs = inputs.squeeze()
    to_pil_image = transforms.ToPILImage()
    img = to_pil_image(inputs.cpu())
    img.show()



def map_generate(attention_map, pred, p1, p2):
    batches, feaC, feaH, feaW = attention_map.size()

    out_map=torch.zeros_like(attention_map.mean(1))

    for batch_index in range(batches):
        map_tpm = attention_map[batch_index]
        map_tpm = map_tpm.reshape(feaC, feaH*feaW)
        map_tpm = map_tpm.permute([1, 0])
        p1_tmp = p1.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p1_tmp)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

        pred_tmp = pred[batch_index]
        pred_ind = pred_tmp.argmax()
        p2_tmp = p2[pred_ind].unsqueeze(1)

        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p2_tmp)
        out_map[batch_index] = map_tpm.reshape(feaH, feaW)

    return out_map
def attention3(attention_map,K,crop_image_num,index,branch,theta=0.5, padding_ratio=0.1):
    attention_map = attention_map.clone()
    device = torch.device("cuda:0")  # 选择一个可用的GPU设备
    # theta = theta.to(device)
    # attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = attention_map.size()

    boxsize=int(imgH/4*3)

    boxsize12 = int(imgH / 2)

    boxsize14 = int(imgH /4)
    size14=int(imgH/4)
    boxcentersize1=int(imgH/2)

    size18 = int(imgH / 8)
    boxcentersize2 = int(imgH /4*3)


    boxcentersize3 = int(imgH)
    if (branch==1):
        scale = imgH // 22
    elif (branch==2):
        scale = imgH // 22
    elif (branch==3):
        scale = imgH // 22
    cropped_images_points_list = []
    boxnum = []
    # cropped_image_points.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
    for batch_index in range(batches):
        cropped_image_points = []
        cropped_image_points.append((0, 0, boxsize, boxsize))
        cropped_image_points.append((0, 0, boxsize12, boxsize12))
        cropped_image_points.append((0, 0, boxsize14, boxsize14))

        # cropped_image_points.append((0, 0, boxsize, boxsize))
        # cropped_image_points.append((size18, size18, size18+boxsize12, size18+boxsize12))
        # cropped_image_points.append((size14, size14, size14+boxsize14, size14+boxsize14))

        cropped_image_points.append((0, imgW - boxsize, boxsize, imgW))
        cropped_image_points.append((0, imgW - boxsize12, boxsize12, imgW ))
        cropped_image_points.append((0, imgW - boxsize14,boxsize14, imgW))

        # cropped_image_points.append((0,imgW-boxsize, boxsize, imgW))
        # cropped_image_points.append((size18, imgW - boxsize12-size18, size18+boxsize12, imgW-size18))
        # cropped_image_points.append((size14, imgW - boxsize14-size14, size14+boxsize14, imgW-size14))

        cropped_image_points.append((imgH - boxsize, 0, imgH, boxsize))
        cropped_image_points.append((imgH - boxsize12, 0, imgH, boxsize12))
        cropped_image_points.append((imgH - boxsize14, 0, imgH, boxsize14))

        # cropped_image_points.append((imgH - boxsize, 0, imgH, boxsize))
        # cropped_image_points.append((imgH - boxsize12-size18, size18, imgH-size18, boxsize12+size18))
        # cropped_image_points.append((imgH - boxsize14-size14, size14, imgH-size14, boxsize14+size14))

        cropped_image_points.append((imgH - boxsize, imgW - boxsize, imgH, imgW))
        cropped_image_points.append((imgH - boxsize12, imgW - boxsize12, imgH, imgW))
        cropped_image_points.append((imgH - boxsize14, imgW - boxsize14, imgH, imgW))

        # cropped_image_points.append((imgH - boxsize, imgW - boxsize, imgH, imgW))
        # cropped_image_points.append((imgH - boxsize12-size18, imgW - boxsize12-size18, imgH-size18, imgW-size18))
        # cropped_image_points.append((imgH - boxsize14-size14, imgW - boxsize14-size14, imgH-size14, imgW-size14))

        cropped_image_points.append((size18,size18, size18+boxsize, size18+boxsize))
        boxnum.append(13)

        # cropped_image_points.append((size14, size14, size14+boxcentersize1, size14+boxcentersize1))
        # cropped_image_points.append((size18, size18, size18+boxcentersize2, size18+boxcentersize2))
        # cropped_image_points.append((0, 0, boxcentersize3, boxcentersize3))
        # boxnum.append(3)

        cropped_images_points_list.append(cropped_image_points)
    return boxnum,cropped_images_points_list
def attention2(attention_map,K,crop_image_num,index,branch,theta=0.5, padding_ratio=0.1):
    attention_map = attention_map.clone()
    device = torch.device("cuda:0")  # 选择一个可用的GPU设备
    # theta = theta.to(device)
    # attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = attention_map.size()

    boxsize=int(imgH/4*3)

    boxsize12 = int(imgH / 2)

    boxsize14 = int(imgH /4)
    size14=int(imgH/4)
    boxcentersize1=int(imgH/2)

    size18 = int(imgH / 8)
    boxcentersize2 = int(imgH /4*3)


    boxcentersize3 = int(imgH)
    if (branch==1):
        scale = imgH // 22
    elif (branch==2):
        scale = imgH // 22
    elif (branch==3):
        scale = imgH // 22
    cropped_images_points_list = []
    boxnum = []
    # cropped_image_points.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
    for batch_index in range(batches):
        cropped_image_points = []
        # cropped_image_points.append((0, 0, boxsize, boxsize))
        # cropped_image_points.append((0, 0, boxsize12, boxsize12))
        # cropped_image_points.append((0, 0, boxsize14, boxsize14))

        cropped_image_points.append((0, 0, boxsize, boxsize))
        cropped_image_points.append((size18, size18, size18+boxsize12, size18+boxsize12))
        cropped_image_points.append((size14, size14, size14+boxsize14, size14+boxsize14))

        # cropped_image_points.append((0, imgW - boxsize, boxsize, imgW))
        # cropped_image_points.append((0, imgW - boxsize12, boxsize12, imgW ))
        # cropped_image_points.append((0, imgW - boxsize14,boxsize14, imgW))

        cropped_image_points.append((0,imgW-boxsize, boxsize, imgW))
        cropped_image_points.append((size18, imgW - boxsize12-size18, size18+boxsize12, imgW-size18))
        cropped_image_points.append((size14, imgW - boxsize14-size14, size14+boxsize14, imgW-size14))

        # cropped_image_points.append((imgH - boxsize, 0, imgH, boxsize))
        # cropped_image_points.append((imgH - boxsize12, 0, imgH, boxsize12))
        # cropped_image_points.append((imgH - boxsize14, 0, imgH, boxsize14))

        cropped_image_points.append((imgH - boxsize, 0, imgH, boxsize))
        cropped_image_points.append((imgH - boxsize12-size18, size18, imgH-size18, boxsize12+size18))
        cropped_image_points.append((imgH - boxsize14-size14, size14, imgH-size14, boxsize14+size14))

        # cropped_image_points.append((imgH - boxsize, imgW - boxsize, imgH, imgW))
        # cropped_image_points.append((imgH - boxsize12, imgW - boxsize12, imgH, imgW))
        # cropped_image_points.append((imgH - boxsize14, imgW - boxsize14, imgH, imgW))

        cropped_image_points.append((imgH - boxsize, imgW - boxsize, imgH, imgW))
        cropped_image_points.append((imgH - boxsize12-size18, imgW - boxsize12-size18, imgH-size18, imgW-size18))
        cropped_image_points.append((imgH - boxsize14-size14, imgW - boxsize14-size14, imgH-size14, imgW-size14))

        cropped_image_points.append((size18,size18, size18+boxsize, size18+boxsize))
        boxnum.append(13)

        # cropped_image_points.append((size14, size14, size14+boxcentersize1, size14+boxcentersize1))
        # cropped_image_points.append((size18, size18, size18+boxcentersize2, size18+boxcentersize2))
        # cropped_image_points.append((0, 0, boxcentersize3, boxcentersize3))
        # boxnum.append(3)

        cropped_images_points_list.append(cropped_image_points)
    return boxnum,cropped_images_points_list

def attention1(attention_map,K,crop_image_num,index,branch,theta=0.5, padding_ratio=0.1):
    attention_map = attention_map.clone()
    device = torch.device("cuda:0")  # 选择一个可用的GPU设备
    # theta = theta.to(device)
    # attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = attention_map.size()

    boxsize=int(imgH/4*3)

    boxsize12 = int(imgH / 2)

    boxsize14 = int(imgH /4)
    size14=int(imgH/4)
    boxcentersize1=int(imgH/2)

    size18 = int(imgH / 8)
    boxcentersize2 = int(imgH /4*3)


    boxcentersize3 = int(imgH)
    if (branch==1):
        scale = imgH // 22
    elif (branch==2):
        scale = imgH // 22
    elif (branch==3):
        scale = imgH // 22
    cropped_images_points_list = []
    boxnum = []
    # cropped_image_points.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
    for batch_index in range(batches):
        cropped_image_points = []
        # cropped_image_points.append((0, 0, boxsize, boxsize))
        # cropped_image_points.append((0, 0, boxsize12, boxsize12))
        # cropped_image_points.append((0, 0, boxsize14, boxsize14))

        cropped_image_points.append((0, 0, boxsize, boxsize))
        cropped_image_points.append((size18, size18, size18+boxsize12, size18+boxsize12))
        cropped_image_points.append((size14, size14, size14+boxsize14, size14+boxsize14))

        # cropped_image_points.append((0, imgW - boxsize, boxsize, imgW))
        # cropped_image_points.append((0, imgW - boxsize12, boxsize12, imgW ))
        # cropped_image_points.append((0, imgW - boxsize14,boxsize14, imgW))

        cropped_image_points.append((0,imgW-boxsize, boxsize, imgW))
        cropped_image_points.append((size18, imgW - boxsize12-size18, size18+boxsize12, imgW-size18))
        cropped_image_points.append((size14, imgW - boxsize14-size14, size14+boxsize14, imgW-size14))

        # cropped_image_points.append((imgH - boxsize, 0, imgH, boxsize))
        # cropped_image_points.append((imgH - boxsize12, 0, imgH, boxsize12))
        # cropped_image_points.append((imgH - boxsize14, 0, imgH, boxsize14))

        cropped_image_points.append((imgH - boxsize, 0, imgH, boxsize))
        cropped_image_points.append((imgH - boxsize12-size18, size18, imgH-size18, boxsize12+size18))
        cropped_image_points.append((imgH - boxsize14-size14, size14, imgH-size14, boxsize14+size14))

        # cropped_image_points.append((imgH - boxsize, imgW - boxsize, imgH, imgW))
        # cropped_image_points.append((imgH - boxsize12, imgW - boxsize12, imgH, imgW))
        # cropped_image_points.append((imgH - boxsize14, imgW - boxsize14, imgH, imgW))

        cropped_image_points.append((imgH - boxsize, imgW - boxsize, imgH, imgW))
        cropped_image_points.append((imgH - boxsize12-size18, imgW - boxsize12-size18, imgH-size18, imgW-size18))
        cropped_image_points.append((imgH - boxsize14-size14, imgW - boxsize14-size14, imgH-size14, imgW-size14))

        cropped_image_points.append((size18,size18, size18+boxsize, size18+boxsize))
        boxnum.append(13)

        # cropped_image_points.append((size14, size14, size14+boxcentersize1, size14+boxcentersize1))
        # cropped_image_points.append((size18, size18, size18+boxcentersize2, size18+boxcentersize2))
        # cropped_image_points.append((0, 0, boxcentersize3, boxcentersize3))
        # boxnum.append(3)

        cropped_images_points_list.append(cropped_image_points)
    return boxnum,cropped_images_points_list

def attention_im(images, attention_map,K,crop_image_num,index,branch,theta=0.5, padding_ratio=0.1):
    images = images.clone()
    device = torch.device("cuda:0")  # 选择一个可用的GPU设备
    # theta = theta.to(device)
    attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = images.size()

    if (branch==1):
        scale = imgH // 22
    elif (branch==2):
        scale = imgH // 22
    elif (branch==3):
        scale = imgH // 22

    cropped_images_points_list = []
    boxnum=[]
    for batch_index in range(batches):
        image_tmp = images[batch_index]
        image_tmps=image_tmp
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)   #[1,1,22,22]

        #不扩充到原图大小，后续是要在特征图像进行裁剪
        map_feature=map_tpm
        map_feature = (map_feature - map_feature.min()) / (map_feature.max() - map_feature.min() + 1e-6)
        map_feature = torch.where(map_feature < torch.tensor(0.5).to(device), torch.tensor(0.0).to(device), map_feature)
        # 使用squeeze方法去除维度为1的维度
        map_feature = map_feature.squeeze()  #[22,22]
        #计算聚类点
        K, cropped_image_points= kmean_cpu(map_feature, K=K,index=index)
        boxnum.append(K)
        cropped_images_points_list.append(cropped_image_points)


        # image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        images[batch_index] = image_tmp
    return images, boxnum,cropped_images_points_list


def AffinityPropagation_cpu(images,K=1,index=None):
    # 将图像转为灰度图
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将二维图像数据转换为一维数据

    image=images.clone()
    imagesizeW=image.shape[0]
    imagesizeH = image.shape[1]
    image = image.cpu()
    image_1d = image.reshape((-1, 1))
    # 创建AffinityPropagation对象
    clustering = AffinityPropagation(damping=0.8, preference=-5)
    # 执行聚类
    clustering.fit(image_1d)
    # 获取聚类结果
    labels = clustering.labels_
    # 计算聚类点的数量
    # num_clusters = len(np.unique(labels))
    cluster_centers = clustering.cluster_centers_
    cluster_centers_indices = clustering.cluster_centers_indices_
    # 将聚类中心在张量中的位置映射为二维坐标
    cluster_center_coords = []
    cropped_image_points = []
    for center_index in cluster_centers_indices:
        row = center_index // imagesizeW
        col = center_index % imagesizeH
        cluster_center_coords.append((col, row))  # 注意此处坐标的顺序是(x, y)
    # num_clusters=len(cluster_center_coords)
    cluster_center_coords = torch.tensor(cluster_center_coords, dtype=torch.float32)

    # 绘制矩形标记聚类点
    if (index == 1):
        box_size = int(imagesizeW /2)  # 矩形框大小
    elif (index == 2):
        box_size = int(imagesizeW /2)  # 矩形框大小
    elif (index == 3):
        box_size = int(imagesizeW /4*3)  # 矩形框大小
    else:
        box_size = int(imagesizeW /2)
    minx = torch.tensor(0.0)
    miny = torch.tensor(0.0)
    maxx = torch.tensor(float(imagesizeW))
    maxy = torch.tensor(float(imagesizeH))
    # 将NumPy数组转换为PyTorch张量
    labels_tensor = torch.from_numpy(labels).to(device)
    # for i in range(num_clusters):
    for center in cluster_center_coords:
        center_x,center_y = center
        #中心扩大
        x_min = torch.max(center_x - box_size / 2, minx)
        y_min = torch.max(center_y - box_size / 2, miny)
        x_max = torch.min(center_x + box_size / 2, maxx)
        y_max = torch.min(center_y + box_size / 2, maxy)
        if x_min > minx and x_max == maxx:
            x_min = torch.tensor(float(imagesizeW - box_size)) #int(imagesizeW - box_size)
        if x_min == minx and x_max < maxx:
            x_max = torch.tensor(float(box_size))
        if y_min == miny and y_max < maxy:
            y_max = torch.tensor(float(box_size))
        if y_min > miny and y_max == maxy:
            y_min = torch.tensor(float(imagesizeH - box_size))
        cropped_image_points.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))



    # cluster_center_coords = sorted(cluster_center_coords, key=lambda point: point[0])
    # cluster_points_tensor = torch.tensor(cluster_center_coords, dtype=torch.float32)
    # cluster_points_tensor = cluster_points_tensor.to(device)


    cropped_image_points = torch.tensor(cropped_image_points, dtype=torch.float32)
    cropped_image_points = torch.unique(cropped_image_points, dim=0)   #去掉重复的框
    # # 对 Tensor 进行排序
    # cropped_image_points, indices = torch.sort(cropped_image_points)
    num_clusters = cropped_image_points.shape[0]

    # for i in range(K - box_num):
    #     x_min =torch.max(torch.tensor(imagesizeW / 2 - box_size / 2), minx) # int(max(imagesizeW / 2 - box_size / 2, 0))
    #     y_min =torch.max(torch.tensor(imagesizeH / 2 - box_size / 2), miny)# int(max(imagesizeH / 2 - box_size / 2, 0))
    #     x_max =torch.min(torch.tensor(imagesizeW / 2 + box_size / 2), maxx)# int(min(imagesizeW / 2 + box_size / 2, imagesizeW))
    #     y_max =torch.min(torch.tensor(imagesizeH / 2 + box_size / 2), maxy)# int(min(imagesizeH / 2 + box_size / 2, imagesizeH))  # 增加中心框
    #     new_data = torch.tensor([x_min, y_min, x_max, y_max])
    #     cropped_image_points = torch.cat((cropped_image_points, new_data.view(1, -1)), dim=0)

    cropped_image_points = cropped_image_points.to(device)
    return num_clusters,cropped_image_points

def kmean_box_inputATT(images,K=1):
    # 将图像转为灰度图
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image=images.clone()
    imagesizeW=image.shape[0]
    imagesizeH = image.shape[1]
    gray=image.cpu()
    flatten_gray = gray.reshape((-1, 1))
    flatten_gray = np.float32(flatten_gray)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(flatten_gray, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 对聚类点进行标记
    centers = np.uint8(centers)
    cluster_points = []
    #存储需要裁切的四个角
    cropped_image_points=[]
    # 绘制矩形标记聚类点
    box_size = int(imagesizeW /4*3)
    # box_size =100  # 矩形框大小
    # 将NumPy数组转换为PyTorch张量
    labels_tensor = torch.from_numpy(labels).to(device)
    minx=torch.tensor(0.0)
    miny = torch.tensor(0.0)
    maxx=torch.tensor(float(imagesizeW))
    maxy = torch.tensor(float(imagesizeH))

    box_min_x = torch.tensor(float(imagesizeW))
    box_min_y = torch.tensor(float(imagesizeW))
    box_max_x=torch.tensor(0.0)
    box_max_y = torch.tensor(0.0)
    for i in range(K):
        cluster_pixels = torch.nonzero(labels_tensor == i).flatten().float()
        if cluster_pixels.size(0) > 0:
            center_x = torch.mean(cluster_pixels % imagesizeW)+80
            center_y = torch.mean(torch.div(cluster_pixels, imagesizeH, rounding_mode='trunc'))+80


            # 将聚类点坐标加入列表
            cluster_points.append((center_x.item(), center_y.item()))

            #中心扩大
            x_min = torch.max(center_x - box_size / 2-10, minx)
            y_min = torch.max(center_y - box_size / 2-10, miny)
            x_max = torch.min(center_x + box_size / 2+10, maxx)
            y_max = torch.min(center_y + box_size / 2+10, maxy)
            if x_min > minx and x_max == maxx:
                x_min = torch.tensor(float(imagesizeW - box_size)) #int(imagesizeW - box_size)
            if x_min == minx and x_max < maxx:
                x_max = torch.tensor(float(box_size))
            if y_min == miny and y_max < maxy:
                y_max = torch.tensor(float(box_size))
            if y_min > miny and y_max == maxy:
                y_min = torch.tensor(float(imagesizeH - box_size))
            # cropped_image_points.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
            if box_min_x >= x_min:
                box_min_x = x_min
            if box_min_y >= y_min:
                box_min_y = y_min
            if box_max_x <= x_max:
                box_max_x = x_max
            if box_max_y <= y_max:
                box_max_y = y_max

    x_min=int(x_min)
    x_max = int(x_max)
    y_min = int(y_min)
    y_max = int(y_max)
    # cluster_points = sorted(cluster_points, key=lambda point: point[0])
    # cluster_points_tensor = torch.tensor(cluster_points, dtype=torch.float32)
    # cluster_points_tensor = cluster_points_tensor.to(device)

    # cropped_image_points.append((box_min_x.item(), box_min_y.item(), box_max_x.item(), box_max_y.item()))
    # cropped_image_points = torch.tensor(cropped_image_points, dtype=torch.float32)
    # cropped_image_points = torch.unique(cropped_image_points, dim=0)   #去掉重复的框
    # 对 Tensor 进行排序
    # cropped_image_points, indices = torch.sort(cropped_image_points)

    # box_num = cropped_image_points.shape[0]

    # for i in range(K - box_num):
    #     x_min =torch.max(torch.tensor(imagesizeW / 2 - box_size / 2), minx) # int(max(imagesizeW / 2 - box_size / 2, 0))
    #     y_min =torch.max(torch.tensor(imagesizeH / 2 - box_size / 2), miny)# int(max(imagesizeH / 2 - box_size / 2, 0))
    #     x_max =torch.min(torch.tensor(imagesizeW / 2 + box_size / 2), maxx)# int(min(imagesizeW / 2 + box_size / 2, imagesizeW))
    #     y_max =torch.min(torch.tensor(imagesizeH / 2 + box_size / 2), maxy)# int(min(imagesizeH / 2 + box_size / 2, imagesizeH))  # 增加中心框
    #     new_data = torch.tensor([x_min, y_min, x_max, y_max])
    #     cropped_image_points = torch.cat((cropped_image_points, new_data.view(1, -1)), dim=0)
    # num_clusters = cropped_image_points.shape[0]
    # cropped_image_points = cropped_image_points.to(device)
    return x_min, y_min, x_max, y_max

def kmean_cpu(images,K=1,index=None):
    # 将图像转为灰度图
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image=images.clone()
    imagesizeW=image.shape[0]
    imagesizeH = image.shape[1]
    gray=image.cpu()
    flatten_gray = gray.reshape((-1, 1))
    flatten_gray = np.float32(flatten_gray)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(flatten_gray, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 对聚类点进行标记
    centers = np.uint8(centers)
    cluster_points = []
    #存储需要裁切的四个角
    cropped_image_points=[]
    # 绘制矩形标记聚类点
    if (index==1):
        box_size =int(imagesizeW/2) # 矩形框大小
    elif(index==2):
        box_size = int(imagesizeW /2) # 矩形框大小
    elif (index == 3):
        box_size = int(imagesizeW /2)  # 矩形框大小
    else:
        box_size = int(imagesizeW /2)
    # box_size =100  # 矩形框大小
    # 将NumPy数组转换为PyTorch张量
    labels_tensor = torch.from_numpy(labels).to(device)
    minx=torch.tensor(0.0)
    miny = torch.tensor(0.0)
    maxx=torch.tensor(float(imagesizeW))
    maxy = torch.tensor(float(imagesizeH))

    box_min_x = torch.tensor(float(imagesizeW))
    box_min_y = torch.tensor(float(imagesizeW))
    box_max_x=torch.tensor(0.0)
    box_max_y = torch.tensor(0.0)
    for i in range(K):
        cluster_pixels = torch.nonzero(labels_tensor == i).flatten().float()
        if cluster_pixels.size(0) > 0:
            center_x = torch.mean(cluster_pixels % imagesizeW)
            center_y = torch.mean(torch.div(cluster_pixels, imagesizeH, rounding_mode='trunc'))

            # if (index == 1):
            #     center_x=center_x+2
            #     center_y=center_y+2
            # elif (index == 2):
            #     center_x = center_x + 2
            #     center_y = center_y+2
            # elif (index == 3):
            #     center_x = center_x+2
            #     center_y = center_y+2
            # else:
            #     center_x = center_x
            #     center_y = center_y
            # 将聚类点坐标加入列表
            cluster_points.append((center_x.item(), center_y.item()))

            #中心扩大
            x_min = torch.max(center_x - box_size / 2, minx)
            y_min = torch.max(center_y - box_size / 2, miny)
            x_max = torch.min(center_x + box_size / 2, maxx)
            y_max = torch.min(center_y + box_size / 2, maxy)
            if x_min > minx and x_max == maxx:
                x_min = torch.tensor(float(imagesizeW - box_size)) #int(imagesizeW - box_size)
            if x_min == minx and x_max < maxx:
                x_max = torch.tensor(float(box_size))
            if y_min == miny and y_max < maxy:
                y_max = torch.tensor(float(box_size))
            if y_min > miny and y_max == maxy:
                y_min = torch.tensor(float(imagesizeH - box_size))
            cropped_image_points.append((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
            if box_min_x >= x_min:
                box_min_x = x_min
            if box_min_y >= y_min:
                box_min_y = y_min
            if box_max_x <= x_max:
                box_max_x = x_max
            if box_max_y <= y_max:
                box_max_y = y_max


    cluster_points = sorted(cluster_points, key=lambda point: point[0])
    cluster_points_tensor = torch.tensor(cluster_points, dtype=torch.float32)
    cluster_points_tensor = cluster_points_tensor.to(device)

    # cropped_image_points.append((box_min_x.item(), box_min_y.item(), box_max_x.item(), box_max_y.item()))
    cropped_image_points = torch.tensor(cropped_image_points, dtype=torch.float32)
    cropped_image_points = torch.unique(cropped_image_points, dim=0)   #去掉重复的框
    # 对 Tensor 进行排序
    cropped_image_points, indices = torch.sort(cropped_image_points)


    num_clusters = cropped_image_points.shape[0]
    cropped_image_points = cropped_image_points.to(device)
    return num_clusters,cropped_image_points

def enhance_contrast(image, importance_tensor, gain=1.0, bias=0.0):
    # 将张量标准化在0-1范围内
    # normalized_tensor = (importance_tensor - importance_tensor.min()) / (importance_tensor.max() - importance_tensor.min())

    # 增强对比度
    enhanced_tensor = importance_tensor * gain + bias
    enhanced_brightness = enhanced_tensor.item()
    # 将张量应用于图像
    enhanced_image = torchvision.transforms.functional.adjust_brightness(image, enhanced_brightness)

    return enhanced_image

def highlight(images, attention_map, attention_map2, attention_map3,K,theta=0.5, padding_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.clone()
    attention_map = attention_map.clone().detach()
    attention_map2 = attention_map2.clone().detach()
    attention_map3 = attention_map3.clone().detach()
    batches, _, imgH, imgW = images.size()
    imageheight=int(imgH / 2)
    imagewidth = int(imgW / 2)
    for batch_index in range(batches):
        image_tmp = images[batch_index]
        image_tmps=image_tmp.unsqueeze(0)
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        # map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = torch.nn.functional.interpolate(map_tpm, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        # map_tpm2 = torch.nn.functional.upsample_bilinear(map_tpm2, size=(imgH, imgW)).squeeze()
        map_tpm2 = torch.nn.functional.interpolate(map_tpm2, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        # map_tpm3 = torch.nn.functional.upsample_bilinear(map_tpm3, size=(imgH, imgW)).squeeze()
        map_tpm3 = torch.nn.functional.interpolate(map_tpm3, size=(imgH, imgW), mode='bilinear', align_corners=True).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        map_tpm = (map_tpm + map_tpm2 + map_tpm3)
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        featuremap=map_tpm
        # featuremap = torch.where(featuremap < torch.tensor(0.2).to(device), torch.tensor(0.0).to(device), featuremap)
        x_min, y_min, x_max, y_max = kmean_box_inputATT(featuremap, K=K)
        image_tmp = image_tmp[:, x_min:y_max, y_min:x_max].unsqueeze(0)
        image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        # flag = torch.rand(1)
        # if flag < (3 / 4):
        #     featuremap = torch.where(featuremap < torch.tensor(0.2).to(device), torch.tensor(0.0).to(device), featuremap)
        #     x_min, y_min, x_max, y_max = kmean_box_inputATT(featuremap, K=K)
        #     image_tmp = image_tmp[:, x_min:y_max, y_min:x_max].unsqueeze(0)
        #     image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        # elif (3 / 4) <= flag < (1):
        #     featuremap = torch.where(featuremap < torch.tensor(0.2).to(device), torch.tensor(0.0).to(device), featuremap)
        #     x_min, y_min, x_max, y_max = kmean_box_inputATT(featuremap, K=K)
        #     image_tmp = image_tmp[:, x_min:y_max, y_min:x_max].unsqueeze(0)
        #     image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear',align_corners=True)
        #     image_tmps = torch.nn.functional.interpolate(image_tmp, size=(imageheight, imagewidth), mode='bilinear',align_corners=True)
        #     # 对张量进行上下翻转
        #     image_tmps_up_right = torch.flip(image_tmps, dims=[2])
        #     # 对张量进行左右翻转
        #     image_tmps_down_left = torch.flip(image_tmps, dims=[3])
        #     image_tmps_down_right = torch.flip(image_tmps_up_right, dims=[3])
        #     y = torch.zeros(1, 3, imgH, imgW)
        #     y[:, :, :imageheight, :imagewidth] = image_tmps
        #     y[:, :, :imageheight, imagewidth:] = image_tmps_down_left
        #     y[:, :, imageheight:, :imagewidth] = image_tmps_up_right
        #     y[:, :, imageheight:, imagewidth:] = image_tmps_down_right
        #     image_tmp=y.squeeze()
        # elif flag >= (2 / 3):
        #     image_tmps = torch.nn.functional.interpolate(image_tmps, size=(imageheight, imagewidth), mode='bilinear',
        #                                                  align_corners=True)
        #     # 对张量进行上下翻转
        #     image_tmps_up_right = torch.flip(image_tmps, dims=[2])
        #     # 对张量进行左右翻转
        #     image_tmps_down_left = torch.flip(image_tmps, dims=[3])
        #     image_tmps_down_right = torch.flip(image_tmps_up_right, dims=[3])
        #     y = torch.zeros(1, 3, imgH, imgW)
        #     y[:, :, :imageheight, :imagewidth] = image_tmps
        #     y[:, :, :imageheight, imagewidth:] = image_tmps_down_right
        #     y[:, :, imageheight:, :imagewidth] = image_tmps_up_right
        #     y[:, :, imageheight:, imagewidth:] = image_tmps_down_left
        #     image_tmp = y.squeeze()



        # map_tpm = torch.where(map_tpm < torch.tensor(0.2).to(device), torch.tensor(0.0).to(device), map_tpm)
        # x_min, y_min, x_max, y_max = kmean_box_inputATT(map_tpm, K=K)
        # mask = (map_tpm == 0)
        # image_tmp[:, mask] = 0  # 将image_tmp中相同位置的元素改为0
        #


        # map_tpm = map_tpm >= theta
        #
        # nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        # height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        # height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        # width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        # width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
        # image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        # image_tmp = image_tmp[:, x_min:y_max, y_min:x_max].unsqueeze(0)

        # image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        images[batch_index] = image_tmp
    return images

def highlight_im(images, attention_map, attention_map2, attention_map3,K, crop_image_num,index,branch,theta=0.5, padding_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.clone()
    attention_map = attention_map.clone().detach()
    attention_map2 = attention_map2.clone().detach()
    attention_map3 = attention_map3.clone().detach()
    batches, _, imgH, imgW = images.size()
    orginsize = imgH * imgW

    if (branch==1):
        scale = imgH // 22
        theta=0.5
    elif (branch==2):
        scale = imgH // 22
        theta = 0.4
    elif (branch==3):
        scale = imgH // 22
        theta = 0.3
    # images_cluster_points = torch.zeros(batches,K, 2)  #5表示有几个聚类点，2表示二维坐标
    # cropped_images_points = torch.cuda.FloatTensor()
    cropped_images_points_list = []
    boxnum=[]
    for batch_index in range(batches):
        image_tmp = images[batch_index]
        image_tmps=image_tmp


        map_tpm1 = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_feature1=map_tpm1
        # map_feature1 = torch.where(map_feature1 < torch.tensor(0.5).to(device), torch.tensor(0.0).to(device), map_feature1)
        # map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm1 = torch.nn.functional.interpolate(map_tpm1, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        map_tpm1 = (map_tpm1 - map_tpm1.min()) / (map_tpm1.max() - map_tpm1.min() + 1e-6)


        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        map_feature2 = map_tpm2
        # map_feature2 = torch.where(map_feature2 < torch.tensor(0.4).to(device), torch.tensor(0.0).to(device),map_feature2)
        # map_tpm2 = torch.nn.functional.upsample_bilinear(map_tpm2, size=(imgH, imgW)).squeeze()
        map_tpm2 = torch.nn.functional.interpolate(map_tpm2, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        map_feature3 = map_tpm3
        # map_feature3 = torch.where(map_feature3 < torch.tensor(0.3).to(device), torch.tensor(0.0).to(device),map_feature3)
        # map_tpm3 = torch.nn.functional.upsample_bilinear(map_tpm3, size=(imgH, imgW)).squeeze()
        map_tpm3 = torch.nn.functional.interpolate(map_tpm3, size=(imgH, imgW), mode='bilinear', align_corners=True).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        map_tpm = (map_tpm1 + map_tpm2 + map_tpm3)
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


        # map_feature = map_tpm
        # map_feature=map_feature.unsqueeze(0).unsqueeze(0)
        # if (index == 1):
        #     # map_feature1 = torch.nn.functional.interpolate(map_feature1, size=(22, 22), mode='bilinear',align_corners=True).squeeze()
        #     map_feature3 = torch.nn.functional.interpolate(map_feature3, size=(22, 22), mode='bilinear',align_corners=True).squeeze()
        # elif (index == 2):
        #     # map_feature2 = torch.nn.functional.interpolate(map_feature2, size=(22, 22), mode='bilinear',align_corners=True).squeeze()
        #     map_feature3 = torch.nn.functional.interpolate(map_feature3, size=(22, 22), mode='bilinear',align_corners=True).squeeze()
        # elif (index == 3):
        #     map_feature1 = torch.nn.functional.interpolate(map_feature1, size=(22, 22), mode='bilinear', align_corners=True).squeeze()
        #     map_feature2 = torch.nn.functional.interpolate(map_feature2, size=(22, 22), mode='bilinear',align_corners=True).squeeze()
        map_feature=(map_feature3+map_feature2+map_feature1)/3
        # map_feature = torch.where(map_feature <torch.tensor(0.4).to(device), torch.tensor(0.0).to(device), map_feature)
        # 使用squeeze方法去除维度为1的维度
        map_feature = map_feature.squeeze()
        # 计算聚类点
        K, cropped_image_points = kmean_cpu(map_feature, K=K, index=index)
        boxnum.append(K)
        cropped_images_points_list.append(cropped_image_points)

        map_tpm = map_tpm >= theta
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)

        # corpsize = image_tmp.shape[2] * image_tmp.shape[3]
        # if corpsize / orginsize > 0.6:
        #     cluster_points, cropped_image_points = kmean_cpu(map_tpm, K=K)
        #     cropped_image_points, max_top, max_right, min_bottom, min_left = boxNMS(cropped_image_points, scale=1)
            # image_tmp = image_tmps[:, min_bottom:max_top, min_left:max_right].unsqueeze(0)
        # image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()
        image_tmp = torch.nn.functional.interpolate(image_tmp, size=(imgH, imgW), mode='bilinear',align_corners=True).squeeze()
        images[batch_index] = image_tmp
        # images_cluster_points[batch_index] = cluster_points


        # images_cluster_points[batch_index] = cluster_points
        # cropped_images_points[batch_index] = cropped_image_points
    # return images,cropped_images_points
    return images,boxnum,cropped_images_points_list


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)



def test(store_name,test_acc_max,net,test_loader,CELoss, batch_size, num_class,test_path,K,crop_image_num,channal):
    exp_dir = 'save_path/' + store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    net.eval()
    # partnet.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    total_loss=0
    total_loss1=0
    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4=0
    correct_com_branch=0
    correct5=0
    correct6 = 0
    correct7 = 0
    correct8 = 0
    correct9 = 0
    correct10 = 0
    correct11 = 0
    correct12 = 0
    correct13 = 0
    correct_ATT = 0
    correct_ATT_part = 0
    correct_part = 0
    total = 0
    idx = 0
    crop_image_total=K*crop_image_num
    size1 = 16
    size2 = 16
    size3 = 16
    channal1 = 256
    channal2 = 256
    channal3 = 256
    part_channal=256
    device = torch.device("cuda")
    # 初始化空的预测结果和真实标签列表
    y_pred_list = []
    y_true_list = []
    class_correct = {}
    class_total = {}
    avg_class_correct = 0
    # 计算每个类别的准确率和召回率
    precisions = []
    recalls = []
    # transform_test = transforms.Compose([
    #     # transforms.Resize((550, 550)),
    #     # transforms.Resize((350, 350)),
    #     # transforms.CenterCrop(448),
    #     # transforms.CenterCrop(340),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # testset = torchvision.datasets.ImageFolder(root=test_path,
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    # test_loader.transform = transform_test
    # CELoss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            idx = batch_idx
            weight_loss = []
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)

            # inputs, targets = torch.tensor(inputs), torch.tensor(targets)
            inputs, targets = Variable(inputs), Variable(targets)
            batches, channels, imgH, imgW = inputs.size()
            if batches == 1:
                continue

            x1_c, x2_c, x3_c, x_c_all, inputs_ATT = net(inputs,
                                                        index=0,
                                                        K=K,
                                                        crop_image_num=crop_image_num,
                                                        isBranch=True)
            # loss = CELoss(x1_c, targets) + \
            #        CELoss(x2_c, targets) + \
            #        CELoss(x3_c, targets) + \
            #        CELoss(x_c_all, targets)
            # part_classifier = x1_c + x2_c + x3_c + x_c_all
            loss = CELoss(x1_c, targets) + \
                   CELoss(x2_c, targets) + \
                   CELoss(x3_c, targets)
            # loss = ce_loss(x1_c, targets, reduction='mean') + \
            #              ce_loss(x2_c, targets, reduction='mean') + \
            #              ce_loss(x3_c, targets, reduction='mean')
            part_classifier = x1_c + x2_c + x3_c
            weight_loss.append(loss)
            total_loss += loss.item()
            inputs1 = inputs_ATT

            part_classifier11, part_classifier11_1, part_classifier11_2, part_classifier11_3, inputsATT1= net(inputs1, index=1,
                                                           K=K,
                                                            crop_image_num=crop_image_num, isBranch=True)

            # loss1 = CELoss(part_classifier11, targets) + \
            #         CELoss(part_classifier11_1, targets) + \
            #         CELoss(part_classifier11_2, targets) + \
            #         CELoss(part_classifier11_3, targets)
            #
            # part_classifier_branch = part_classifier11 + part_classifier11_1 + part_classifier11_2 + part_classifier11_3
            loss1 =CELoss(part_classifier11_1, targets) + \
                    CELoss(part_classifier11_2, targets) + \
                    CELoss(part_classifier11_3, targets)
            # loss1 = ce_loss(part_classifier11_1, targets, reduction='mean') + \
            #        ce_loss(part_classifier11_2, targets, reduction='mean') + \
            #        ce_loss(part_classifier11_3, targets, reduction='mean')
            part_classifier_branch =part_classifier11_1 + part_classifier11_2 + part_classifier11_3

            # loss1 = CELoss(part_classifier11, targets)

            total_loss1+=loss1.item()
            weight_loss.append(loss1)


            # inputs2 = inputs_ATT
            #
            # part_classifier22, part_classifier22_1, part_classifier22_2, part_classifier22_3, inputsATT2= net(inputs2,index=2,K=K,crop_image_num=crop_image_num,isBranch=True)
            # loss2 = CELoss(part_classifier22, targets) + \
            #         CELoss(part_classifier22_1, targets) + \
            #         CELoss(part_classifier22_2, targets) + \
            #         CELoss(part_classifier22_3, targets)
            # # loss2 = CELoss(part_classifier22, targets)
            #
            # part_classifier22 = part_classifier22 + part_classifier22_1 + part_classifier22_2 + part_classifier22_3
            # weight_loss.append(loss2)



            # 训练网络，产生特征图
            # inputs3 = inputs_ATT
            # part_classifier33, part_classifier33_1, part_classifier33_2, part_classifier33_3, inputs_ATT3= net(inputs3, index=3,
            #                                                  K=K,
            #                                                 crop_image_num=crop_image_num, isBranch=True)
            # # [16,100],[16,100],[16,100]   [16,1024,16,16]    [16,1024,16,16]   [16,1024,8,8]
            # loss3 = CELoss(part_classifier33, targets) + \
            #         CELoss(part_classifier33_1, targets) + \
            #         CELoss(part_classifier33_2, targets) + \
            #         CELoss(part_classifier33_3, targets)
            #
            # part_classifier33 = part_classifier33 + part_classifier33_1 + part_classifier33_2 + part_classifier33_3
            # # loss3 = CELoss(part_classifier33, targets)
            # weight_loss.append(loss3)

            weight_loss = torch.tensor(weight_loss, dtype=torch.float32)
            hidden_weight = (1 - (weight_loss - torch.min(weight_loss)) / (torch.max(weight_loss) - torch.min(weight_loss))) + torch.mean(weight_loss)
            # output_all =part_classifier* hidden_weight[0]+ part_classifier11 * hidden_weight[1]+part_classifier22 * hidden_weight[2]+part_classifier33 * hidden_weight[3]
            output_all=part_classifier* hidden_weight[0]+part_classifier_branch * hidden_weight[1]

            #概率分布
            x1_c_logits = F.softmax(x1_c, dim=1)
            x2_c_logits = F.softmax(x2_c, dim=1)
            x3_c_logits = F.softmax(x3_c, dim=1)
            x_c_all_logits = F.softmax(x_c_all, dim=1)
            part_classifier_logits = F.softmax(part_classifier, dim=1)
            part_classifier11_logits = F.softmax(part_classifier11, dim=1)
            part_classifier11_1_logits = F.softmax(part_classifier11_1, dim=1)
            part_classifier11_2_logits = F.softmax(part_classifier11_2, dim=1)
            part_classifier11_3_logits = F.softmax(part_classifier11_3, dim=1)
            part_classifier_branch_logits = F.softmax(part_classifier_branch, dim=1)
            part_classifier_all = part_classifier + part_classifier_branch
            part_classifier_all_logits = F.softmax(part_classifier_all, dim=1)
            part_classifier_all_weight_logits = F.softmax(output_all, dim=1)

            _, predicted_labels = torch.max(part_classifier_all_logits, dim=-1)
            # 将预测结果和真实标签添加到列表中
            y_pred_list.append(part_classifier_all_logits)
            y_true_list.append(targets)

            # 统计每个类别的准确率
            for i in range(len(targets)):
                target = targets[i].item()
                predicted_label = predicted_labels[i].item()
                if target == predicted_label:
                    # 预测正确
                    class_correct[target] = class_correct.get(target, 0) + 1
                class_total[target] = class_total.get(target, 0) + 1

            #获取下标
            predicted_x1_c=torch.argmax(x1_c_logits, dim=-1)
            predicted_x2_c = torch.argmax(x2_c_logits, dim=-1)
            predicted_x3_c = torch.argmax(x3_c_logits, dim=-1)
            predicted_x_c_all = torch.argmax(x_c_all_logits, dim=-1)
            predicted_part_classifier = torch.argmax(part_classifier_logits, dim=-1)
            predicted_part_classifier11 = torch.argmax(part_classifier11_logits, dim=-1)
            predicted_part_classifier11_1 = torch.argmax(part_classifier11_1_logits, dim=-1)
            predicted_part_classifier11_2 = torch.argmax(part_classifier11_2_logits, dim=-1)
            predicted_part_classifier11_3 = torch.argmax(part_classifier11_3_logits, dim=-1)
            predicted_part_classifier_branch = torch.argmax(part_classifier_branch_logits, dim=-1)
            predicted_part_part_classifier_all = torch.argmax(part_classifier_all_logits, dim=-1)
            predicted_part_part_classifier_all_weight = torch.argmax(part_classifier_all_weight_logits, dim=-1) #[batchsize]
            predicted_part_vote=voting(predicted_x2_c, predicted_x3_c, predicted_part_classifier,predicted_part_classifier11_2,predicted_part_classifier11_3,predicted_part_classifier_branch,predicted_part_part_classifier_all,predicted_part_part_classifier_all_weight,num_classes=num_class)
            predicted_part_vote=predicted_part_vote.to(device)
            # predicted_part_vote = voting(vote_list)



            test_loss += loss.item() +loss1.item() #4  10  11   12

            total += targets.size(0)
            correct0 += predicted_x1_c.eq(targets.data).cpu().sum()
            correct1 += predicted_x2_c.eq(targets.data).cpu().sum()
            correct2 += predicted_x3_c.eq(targets.data).cpu().sum()
            correct3 += predicted_x_c_all.eq(targets.data).cpu().sum()
            correct4 += predicted_part_classifier.eq(targets.data).cpu().sum()
            correct5 += predicted_part_classifier11.eq(targets.data).cpu().sum()
            correct6 += predicted_part_classifier11_1.eq(targets.data).cpu().sum()
            correct7 += predicted_part_classifier11_2.eq(targets.data).cpu().sum()
            correct8 += predicted_part_classifier11_3.eq(targets.data).cpu().sum()
            correct9 += predicted_part_classifier_branch.eq(targets.data).cpu().sum()
            correct10 += predicted_part_part_classifier_all.eq(targets.data).cpu().sum()
            correct11 += predicted_part_part_classifier_all_weight.eq(targets.data).cpu().sum()
            correct12 += predicted_part_vote.eq(targets.data).cpu().sum()
            # correct13 += predicted6.eq(targets.data).cpu().sum()



            if batch_idx % 50 == 0:
                print('Step: %d  | Acc0: %.3f%% | Acc1: %.3f%%  | Acc2: %.3f%% | Acc3: %.3f%%  | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%% | Acc7: %.3f%%  | Acc8: %.3f%% | Acc9: %.3f%%  | Acc10: %.3f%% | Acc11: %.3f%% | Acc12: %.3f%% | tatal:%d' % (
                batch_idx, 100. * float(correct0) / total,
                100. * float(correct1) / total,
                100. * float(correct2) / total,
                100. * float(correct3) / total,
                100. * float(correct4) / total,
                100. * float(correct5) / total,
                100. * float(correct6) / total,
                100. * float(correct7) / total,
                100. * float(correct8) / total,
                100. * float(correct9) / total,
                100. * float(correct10) / total,
                100. * float(correct11) / total,
                100. * float(correct12) / total,
                total)
                )
                logging.info(
                    'Step: %d  | Acc0: %.3f%% | Acc1: %.3f%%  | Acc2: %.3f%% | Acc3: %.3f%%  | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%% | Acc7: %.3f%%  | Acc8: %.3f%% | Acc9: %.3f%%  | Acc10: %.3f%% | Acc11: %.3f%% | Acc12: %.3f%% | tatal:%d' % (
                        batch_idx, 100. * float(correct0) / total,
                        100. * float(correct1) / total,
                        100. * float(correct2) / total,
                        100. * float(correct3) / total,
                        100. * float(correct4) / total,
                        100. * float(correct5) / total,
                        100. * float(correct6) / total,
                        100. * float(correct7) / total,
                        100. * float(correct8) / total,
                        100. * float(correct9) / total,
                        100. * float(correct10) / total,
                        100. * float(correct11) / total,
                        100. * float(correct12) / total,
                        total)
                    )
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write(
                    'Iteration %d | test_acc = %.5f | test_loss = %.5f | Loss1: %.5f | Loss2: %.5f \n' % (
                        idx, 100. * float(correct11) / total, test_loss, total_loss / (idx + 1), total_loss1 / (idx + 1),
                        ))

    print(
        'Step: %d  | Acc0: %.3f%% | Acc1: %.3f%%  | Acc2: %.3f%% | Acc3: %.3f%%  | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%% | Acc7: %.3f%%  | Acc8: %.3f%% | Acc9: %.3f%%  | Acc10: %.3f%% | Acc11: %.3f%% | Acc12: %.3f%% | tatal:%d' % (
            idx, 100. * float(correct0) / total,
            100. * float(correct1) / total,
            100. * float(correct2) / total,
            100. * float(correct3) / total,
            100. * float(correct4) / total,
            100. * float(correct5) / total,
            100. * float(correct6) / total,
            100. * float(correct7) / total,
            100. * float(correct8) / total,
            100. * float(correct9) / total,
            100. * float(correct10) / total,
            100. * float(correct11) / total,
            100. * float(correct12) / total,
            total)
        )
    logging.info(
        'Step: %d  | Acc0: %.3f%% | Acc1: %.3f%%  | Acc2: %.3f%% | Acc3: %.3f%%  | Acc4: %.3f%% | Acc5: %.3f%% | Acc6: %.3f%% | Acc7: %.3f%%  | Acc8: %.3f%% | Acc9: %.3f%%  | Acc10: %.3f%% | Acc11: %.3f%% | Acc12: %.3f%% | tatal:%d' % (
            idx, 100. * float(correct0) / total,
            100. * float(correct1) / total,
            100. * float(correct2) / total,
            100. * float(correct3) / total,
            100. * float(correct4) / total,
            100. * float(correct5) / total,
            100. * float(correct6) / total,
            100. * float(correct7) / total,
            100. * float(correct8) / total,
            100. * float(correct9) / total,
            100. * float(correct10) / total,
            100. * float(correct11) / total,
            100. * float(correct12) / total,
            total)
    )
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write(
            'Iteration %d | test_acc = %.5f | test_loss = %.5f | Loss1: %.5f | Loss2: %.5f \n' % (
                idx, 100. * float(correct11) / total, test_loss, total_loss / (idx + 1), total_loss1 / (idx + 1),
            ))

    test_acc = 100. * float(correct11) / total
    test_loss = test_loss / (idx + 1)

    accuracy_dict = {}
    # 计算每个类别的准确率
    # for cls in class_total:
    #     accuracy = 100.0 * class_correct[cls] / class_total[cls]
    #     accuracy_dict[cls] = accuracy
    #     avg_class_correct = avg_class_correct + accuracy
    #     print('Accuracy of class {} : {:.2f}%'.format(cls, accuracy))
    # avg_class_correct = avg_class_correct / num_class

    # 将预测结果和真实标签列表转换为数组
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    # 将张量转换为 NumPy 数组
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    # 计算每个类别的AP值和计算平均mAP
    ap_values = []

    for i in range(num_class):
        ap = average_precision_score(y_true == i, y_pred[:, i])
        ap_values.append(ap)
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred[:, i])
        # plt.plot(recall, precision, label=f'Class {i}')
        precisions.append(precision)
        recalls.append(recall)
    # 计算mAP值
    avg_accuracy = np.mean(ap_values)
    # print("AP values:", ap_values)
    print("avg_accuracy:", avg_accuracy)

    if (test_acc > test_acc_max):
        # 保存 y_pred
        np.save('save_path/'+store_name+'/our_y_pred_' + store_name + '.npy', y_pred)
        # 保存 y_true
        np.save('save_path/'+store_name+'/our_y_true_' + store_name + '.npy', y_true)
        # 计算每个类别的AP值和计算平均mAP
        ap_values = []

        for i in range(num_class):
            ap = average_precision_score(y_true == i, y_pred[:, i])
            ap_values.append(ap)
            precision, recall, _ = precision_recall_curve(y_true == i, y_pred[:, i])
            # plt.plot(recall, precision, label=f'Class {i}')
            precisions.append(precision)
            recalls.append(recall)
        # 计算mAP值
        mAP = np.mean(ap_values)
        print("AP values:", ap_values)
        print("mAP:", mAP)



    return test_acc, test_loss


def test_tresnetl(net, criterion, batch_size, test_path):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    correct_com2 = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Scale((421, 421)),
        transforms.CenterCrop(368),

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path,
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

        p1 = net.state_dict()['classifier3.1.weight']
        p2 = net.state_dict()['classifier3.4.weight']
        att_map_3 = map_generate(map3, output_3, p1, p2)

        p1 = net.state_dict()['classifier2.1.weight']
        p2 = net.state_dict()['classifier2.4.weight']
        att_map_2 = map_generate(map2, output_2, p1, p2)

        p1 = net.state_dict()['classifier1.1.weight']
        p2 = net.state_dict()['classifier1.4.weight']
        att_map_1 = map_generate(map1, output_1, p1, p2)

        inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
        output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

        outputs_com2 = output_1 + output_2 + output_3 + output_concat
        outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        _, predicted_com2 = torch.max(outputs_com2.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()
        correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

        print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1),
            100. * float(correct_com) / total, correct_com, total))

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss


def loadTrainData_Semi_supervised(batch_size, train_path):
    transform_train = transforms.Compose([
        transforms.Resize((350, 350)),
        # transforms.CenterCrop(340),
        transforms.RandomResizedCrop(340, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)

    # # 根据标签类别进行分层划分
    labels = trainSet.targets
    # labels = torch.LongTensor(trainSet.targets)
    labeled_indices, unlabeled_indices = train_test_split(range(len(trainSet)), stratify=labels, test_size=0.9,random_state=42)


    # 创建标签数据集和无标签数据集
    labeled_dataset = torch.utils.data.Subset(trainSet, labeled_indices)
    unlabeled_dataset = torch.utils.data.Subset(trainSet, unlabeled_indices)

    labeled_dataset = torch.utils.data.ConcatDataset([labeled_dataset])
    unlabeled_dataset = torch.utils.data.ConcatDataset([unlabeled_dataset])

    # 创建数据加载器
    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)     #其中有所有图片，只不过有相应的索引，根据索引可以分类标签和无标签
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    return labeled_dataset,labeled_loader,unlabeled_dataset,unlabeled_loader


def mixup(image1, label1, image2, label2, mixup_prob):
    if random.random() < mixup_prob:
        alpha =0.2# random.uniform(0.0, 1.0)
        mixed_image = alpha * image1 + (1 - alpha) * image2
        mixed_label =label2# alpha * label1 + (1 - alpha) * label2
        return mixed_image, mixed_label
    else:
        return image1, label1

def cutmix(image1, label1, image2, label2, cutmix_prob):
    if random.random() < cutmix_prob:
        beta = random.uniform(0.0, 1.0)
        mask = np.ones(image1.size, dtype=np.float32)
        cx = np.random.randint(image1.size[0])
        cy = np.random.randint(image1.size[1])
        cw = np.random.randint(image1.size[0] // 2, image1.size[0])
        ch = np.random.randint(image1.size[1] // 2, image1.size[1])
        mask[cx-cw//2:cx+cw//2, cy-ch//2:cy+ch//2] = beta
        mixed_image = image1 * mask + image2 * (1 - mask)
        mixed_label = label1 * beta + label2 * (1 - beta)
        return mixed_image, mixed_label
    else:
        return image1, label1

# def loadTrainData(batch_size, train_path, mixup_prob=0.5, cutmix_prob=0.5):
#     transform_train = transforms.Compose([
#         transforms.RandomRotation(degrees=30),
#         transforms.Resize((350, 350)),
#         transforms.CenterCrop(340),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
#     images, labels = [], []
#     for i in range(len(trainSet)):
#         image, label = trainSet[i]
#         images.append(image)
#         labels.append(label)
#
#     mixed_images, mixed_labels = [], []
#     for i in range(len(trainSet)):
#         image1, label1 = images[i], labels[i]
#         image2, label2 = random.choice(images), random.choice(labels)
#
#         mixed_image, mixed_label = mixup(image1, label1, image2, label2, mixup_prob)
#         # mixed_image, mixed_label = cutmix(mixed_image, mixed_label, image2, label2, cutmix_prob)
#
#         mixed_images.append(mixed_image)
#         mixed_labels.append(mixed_label)
#
#     trainSet = torch.utils.data.TensorDataset(torch.stack(mixed_images), torch.stack(mixed_labels))
#     trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=2)
#     return trainLoader

def loadTrainData(batch_size, train_path):
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.Resize((350, 350)),
        transforms.RandomResizedCrop(340, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train,loader=TiffImagePlugin.TiffImageFile)
    # trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train,loader=TiffImagePlugin.TiffImageFile)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainLoader

import numpy as np

def loadData2(batch_size, data_path, train_ratio=0.6, test_ratio=0.3, val_ratio=0.1, labeled_ratio=0.1):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform_train)

    labels = full_dataset.targets
    # 划分训练集和剩余的数据
    train_idx, rest_idx = train_test_split(range(len(full_dataset)), train_size=train_ratio, stratify=labels)
    rest_idx = np.array(rest_idx) # 转化成 numpy 数组
    train_idx = np.array(train_idx)  # 转化成 numpy 数组
    rest_labels = [labels[i] for i in rest_idx]

    # 划分验证集和测试集
    val_idx, test_idx = train_test_split(range(len(rest_labels)), train_size=val_ratio / (val_ratio + test_ratio),
                                         stratify=rest_labels)

    # 划分有标签和无标签的训练集
    labeled_idx, unlabeled_idx = train_test_split(train_idx, train_size=labeled_ratio, stratify=labels[train_idx])

    labeled_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path, transform=transform_train),
        sampler=torch.utils.data.SubsetRandomSampler(labeled_idx),
        batch_size=batch_size, num_workers=16, pin_memory=True,
    )

    unlabeled_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path, transform=transform_train),
        sampler=torch.utils.data.SubsetRandomSampler(unlabeled_idx),
        batch_size=batch_size, num_workers=16, pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path, transform=transform_val_test),
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
        batch_size=batch_size, num_workers=16, pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(data_path, transform=transform_val_test),
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        batch_size=batch_size, num_workers=16, pin_memory=True,
    )

    return labeled_loader, unlabeled_loader, val_loader, test_loader

def loadData(batch_size, path, train_ratio=0.6, test_ratio=0.3, val_ratio=0.1,labeled_ratio=0.1):
    # 定义数据的转换操作
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.Resize((350, 350)),
        transforms.RandomResizedCrop(340, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 创建 StratifiedShuffleSplit 对象
    # split = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio)

    # 加载整个数据集
    # full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform_train, loader=TiffImagePlugin.TiffImageFile)
    full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform_train,loader=TiffImagePlugin.TiffImageFile)
    labels = full_dataset.targets

    # 按比例划分训练集和测试集
    trainindices, testindices, _, _ = train_test_split(range(len(full_dataset)),labels,train_size=train_ratio,stratify=labels,random_state=42)
    trainindices=sorted(trainindices)
    testindices = sorted(testindices)
    trainlabels = [labels[i] for i in trainindices]
    testlabels = [labels[i] for i in testindices]



    # full_dataset_indices = list(range(len(full_dataset)))
    # test_dataset = torch.utils.data.Subset(full_dataset, testindices)

    labels_indices, unlabels_indices, _, _ = train_test_split(
        trainindices,
        [labels[i] for i in trainindices],
        train_size=labeled_ratio,
        stratify=[labels[i] for i in trainindices],
        random_state=42
    )
    labels_indices = sorted(labels_indices)
    unlabels_indices = sorted(unlabels_indices)

    labels_labels = [labels[i] for i in labels_indices]
    unlabels_labels = [labels[i] for i in unlabels_indices]

    test_indices, val_indices, _, _ = train_test_split(
        testindices,
        [labels[i] for i in testindices],
        train_size=test_ratio / (val_ratio + test_ratio),
        stratify=[labels[i] for i in testindices],
        random_state=42
    )


    test_indices = sorted(test_indices)
    val_indices = sorted(val_indices)

    test_labels = [labels[i] for i in test_indices]
    val_labels = [labels[i] for i in val_indices]

    labels_dataset = torch.utils.data.Subset(full_dataset, labels_indices)
    unlabels_dataset = torch.utils.data.Subset(full_dataset, unlabels_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    # 创建数据加载器
    labeled_loader = torch.utils.data.DataLoader(labels_dataset, batch_size=batch_size,shuffle=False)  # 其中有所有图片，只不过有相应的索引，根据索引可以分类标签和无标签
    unlabeled_loader = torch.utils.data.DataLoader(unlabels_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return labels_dataset,labeled_loader, unlabels_dataset,unlabeled_loader,val_dataset, val_loader, test_dataset,test_loader



# 定义初始化函数
def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)

def loadData4(batch_size, path,num_samples_per_class, train_ratio=0.6, test_ratio=0.3, val_ratio=0.1,labeled_ratio=0.1):
    # 定义数据的转换操作
    transform_train = transforms.Compose([
        transforms.Resize((310, 310)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 指定随机种子
    torch.manual_seed(0)
    # 创建 StratifiedShuffleSplit 对象
    # split = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio)

    # 加载整个数据集
    # full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform_train, loader=TiffImagePlugin.TiffImageFile)
    full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform_train)
    labels = full_dataset.targets





    # 按比例划分训练集和测试集
    trainindices, testindices, _, _ = train_test_split(range(len(full_dataset)),labels,train_size=train_ratio,stratify=labels)#random_state=42
    # trainindices=sorted(trainindices)
    # testindices = sorted(testindices)
    trainlabels = [labels[i] for i in trainindices]
    testlabels = [labels[i] for i in testindices]



    # full_dataset_indices = list(range(len(full_dataset)))
    # test_dataset = torch.utils.data.Subset(full_dataset, testindices)

    labels_indices, unlabels_indices, _, _ = train_test_split(
        trainindices,
        [labels[i] for i in trainindices],
        train_size=labeled_ratio,
        stratify=[labels[i] for i in trainindices]
    )
    # 统计每个类别的样本数量
    class_counts = {}
    for label in [labels[i] for i in trainindices]:
        class_counts[label] = class_counts.get(label, 0) + 1

    # 每个类别需要的样本数量
    num_samples_per_class = num_samples_per_class

    # 用于存储选中的样本索引
    labels_indices = []
    unlabels_indices = []
    # 遍历每个类别，选择固定数量的样本
    for label, count in class_counts.items():
        # 获取当前类别的样本索引
        indices = [trainindices[i] for i in range(len(trainindices)) if trainlabels[i] == label]

        # 判断当前类别的样本数量是否符合要求
        if count < num_samples_per_class:
            # 如果样本数量小于指定数量，则全部加入训练集
            labels_indices += indices
        else:
            # 否则从当前类别中随机选择指定数量的样本
            selected_indices = random.sample(indices, num_samples_per_class)
            labels_indices += selected_indices
            unlabels_indices += [i for i in indices if i not in selected_indices]

    # labels_indices = sorted(labels_indices)
    # unlabels_indices = sorted(unlabels_indices)
    random.shuffle(unlabels_indices)
    labels_labels = [labels[i] for i in labels_indices]
    unlabels_labels = [labels[i] for i in unlabels_indices]

    test_indices, val_indices, _, _ = train_test_split(
        testindices,
        [labels[i] for i in testindices],
        train_size=test_ratio / (val_ratio + test_ratio),
        stratify=[labels[i] for i in testindices]
    )

    random.shuffle(test_indices)
    random.shuffle(val_indices)
    
    
    # test_indices = sorted(test_indices)
    # val_indices = sorted(val_indices)

    test_labels = [labels[i] for i in test_indices]
    val_labels = [labels[i] for i in val_indices]

    labels_dataset = torch.utils.data.Subset(full_dataset, labels_indices)
    unlabels_dataset = torch.utils.data.Subset(full_dataset, unlabels_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    # 创建数据加载器
    labeled_loader = torch.utils.data.DataLoader(labels_dataset, batch_size=batch_size,shuffle=True)  # 其中有所有图片，只不过有相应的索引，根据索引可以分类标签和无标签
    unlabeled_loader = torch.utils.data.DataLoader(unlabels_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return full_dataset,labels_dataset,labeled_loader, unlabels_dataset,unlabeled_loader,val_dataset, val_loader, test_dataset,test_loader




def loadData3(batch_size, path, train_ratio=0.6, test_ratio=0.3, val_ratio=0.1,labeled_ratio=0.1):
    # 定义数据的转换操作
    transform_train = transforms.Compose([
        # transforms.RandomRotation(degrees=30),
        transforms.Resize((310, 310)),
        # transforms.RandomResizedCrop(310, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 创建 StratifiedShuffleSplit 对象
    # split = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio)

    # 加载整个数据集
    # full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform_train, loader=TiffImagePlugin.TiffImageFile)
    full_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform_train,loader=TiffImagePlugin.TiffImageFile)
    labels = full_dataset.targets

    # 按比例划分训练集和测试集
    trainindices, testindices, _, _ = train_test_split(range(len(full_dataset)),labels,train_size=train_ratio,stratify=labels,random_state=42)
    trainindices=sorted(trainindices)
    testindices = sorted(testindices)
    trainlabels = [labels[i] for i in trainindices]
    testlabels = [labels[i] for i in testindices]


    # full_dataset_indices = list(range(len(full_dataset)))
    # test_dataset = torch.utils.data.Subset(full_dataset, testindices)

    # labels_indices, unlabels_indices, _, _ = train_test_split(
    #     trainindices,
    #     [labels[i] for i in trainindices],
    #     train_size=labeled_ratio,
    #     stratify=[labels[i] for i in trainindices],
    #     random_state=42
    # )
    # labels_indices = sorted(labels_indices)
    # unlabels_indices = sorted(unlabels_indices)
    #
    # labels_labels = [labels[i] for i in labels_indices]
    # unlabels_labels = [labels[i] for i in unlabels_indices]

    test_indices, val_indices, _, _ = train_test_split(
        testindices,
        [labels[i] for i in testindices],
        train_size=test_ratio / (val_ratio + test_ratio),
        stratify=[labels[i] for i in testindices],
        random_state=42
    )


    test_indices = sorted(test_indices)
    val_indices = sorted(val_indices)

    test_labels = [labels[i] for i in test_indices]
    val_labels = [labels[i] for i in val_indices]

    # labels_dataset = torch.utils.data.Subset(full_dataset, labels_indices)
    # unlabels_dataset = torch.utils.data.Subset(full_dataset, unlabels_indices)
    train_dataset = torch.utils.data.Subset(full_dataset, trainindices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    # 创建数据加载器
    # labeled_loader = torch.utils.data.DataLoader(labels_dataset, batch_size=batch_size,shuffle=False)  # 其中有所有图片，只不过有相应的索引，根据索引可以分类标签和无标签
    # unlabeled_loader = torch.utils.data.DataLoader(unlabels_dataset, batch_size=batch_size, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return full_dataset,train_dataset ,train_loader,val_dataset, val_loader, test_dataset,test_loader


class Cutout(object):
    def __init__(self, mask_size):
        self.mask_size = mask_size

    def __call__(self, image):
        image = np.array(image)
        h, w, _ = image.shape

        mask_value = image.mean(axis=(0, 1))
        top = np.random.randint(0, h - self.mask_size)
        left = np.random.randint(0, w - self.mask_size)
        bottom = top + self.mask_size
        right = left + self.mask_size

        image[top:bottom, left:right, :] = mask_value

        image = Image.fromarray(image)
        return image






def boxshowimage(imgcolor_np, cropped_images_points3, crop_image_total,index):
    img = Image.fromarray(imgcolor_np)
    draw = ImageDraw.Draw(img)

    width = img.size[0]
    if (index==1):
        scale = width // 22
    elif (index==2):
        scale = width // 22
    elif (index==3):
        scale = width // 22
    cropped_images_points3 = cropped_images_points3 * scale
    cropped_images_points3 = cropped_images_points3[0]

    for i in range(crop_image_total):
        x_min, y_min, x_max, y_max = cropped_images_points3[i]
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, i*50))

    img.show(title=str(index))
    return np.array(img)



# def boxNMS(cropped_image_points):
#
#
#     return cropped_image_points

def boxNMS(cropped_image_points,scale):
    device = torch.device('cuda')
    n = cropped_image_points.shape[0]
    # crossed_boxes = []
    crossed_boxes = torch.empty((0, 4), device=device, dtype=torch.float32)  # 在GPU上创建一个空的张量
    # crossed_areas = []  # 用于存储交叉框的面积值
    # cropped_image_points = torch.tensor(cropped_image_points, dtype=torch.float32)
    # x_min, y_min, x_max, y_max
    #前三个两两交叉
    for i in range(n-2):
        box_i = cropped_image_points[i]
        left_i,bottom_i,top_i,right_i = box_i
        for j in range(i+1, n-2):
            box_j = cropped_image_points[j]
            left_j,bottom_j,top_j,right_j = box_j
            crossed_top = torch.min(top_i, top_j)
            crossed_right = torch.min(right_i, right_j)
            crossed_bottom = torch.max(bottom_i, bottom_j)
            crossed_left = torch.max(left_i, left_j)

            if crossed_top>crossed_bottom and crossed_left < crossed_right:
                # crossed_boxes.append((crossed_left, crossed_bottom, crossed_right, crossed_top))
                crossed_boxes = torch.cat((crossed_boxes,torch.tensor([(crossed_left, crossed_bottom, crossed_right, crossed_top)],dtype=torch.float32, device=device)))
                # crossed_area = (crossed_right - crossed_left) * (crossed_top - crossed_bottom)
                # crossed_areas.append(crossed_area)
    #前三个共同交叉，形成一个
    first_3_boxes = cropped_image_points[:3]
    crossed_top = torch.min(first_3_boxes[:,2])
    crossed_right = torch.min(first_3_boxes[:,3])
    crossed_bottom = torch.max(first_3_boxes[:,1])
    crossed_left = torch.max(first_3_boxes[:,0])
    if crossed_top > crossed_bottom and crossed_left < crossed_right:
        # crossed_boxes.append((crossed_left, crossed_bottom, crossed_right, crossed_top))
        crossed_boxes = torch.cat((crossed_boxes,torch.tensor([(crossed_left, crossed_bottom, crossed_right, crossed_top)],dtype=torch.float32, device=device)))
        crossed_area = (crossed_right - crossed_left) * (crossed_top - crossed_bottom)
        # crossed_areas.append(crossed_area)
    #最后一个大概率是一个中心框
    for j in range(n-1, n):
        box_j = cropped_image_points[j]
        left_j, bottom_j, top_j, right_j = box_j
        # crossed_boxes.append((left_j, bottom_j, top_j, right_j))
        crossed_boxes = torch.cat((crossed_boxes,torch.tensor([(crossed_left, crossed_bottom, crossed_right, crossed_top)],dtype=torch.float32, device=device)))
        crossed_area = (crossed_right - crossed_left) * (crossed_top - crossed_bottom)
        # crossed_areas.append(crossed_area)
    # crossed_boxes = torch.tensor(crossed_boxes, dtype=torch.float32)
    # crossed_boxes = crossed_boxes.to(device)

    # crossed_areas_tensor = torch.tensor(crossed_areas, dtype=torch.float32, device=device)
    max_top = int(torch.max(crossed_boxes[:,2])*scale)
    max_right = int(torch.max(crossed_boxes[:,3])*scale)
    min_bottom = int(torch.min(crossed_boxes[:,1])*scale)
    min_left = int(torch.min(crossed_boxes[:,0])*scale)
    # print(crossed_boxes)
    return crossed_boxes,max_top,max_right,min_bottom,min_left

def voting(*predictions,num_classes):
    num_models = len(predictions)
    batch_size = predictions[0].shape[0]
    # num_classes = predictions[0].shape[1]
    # predictions=predictions.cpu()
    # data_array = np.array(predictions)
    # result = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=data_array)
    # # 创建一个张量，用于累计每个类别的票数
    votes = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        for prediction in predictions:
            # for j in num_classes:
            votes[i, prediction[i]] += 1
    _, final_labels = votes.max(dim=1)

    #
    # # # 对每个预测结果进行投票
    # # for prediction in predictions:
    # #     _, predicted_labels = prediction.max(dim=1)
    # #     for i in range(batch_size):
    # #         votes[i, predicted_labels[i]] += 1
    #
    # # 返回得票最多的类别作为最终预测结果
    # _, final_labels = votes.max(dim=1)
    return final_labels


def find_most_common(data):
    """
    找到一组数据中重复率最高的一个

    参数：
    data -- 一个列表或元组，其中包含任意数量的数据

    返回值：
    最常见的数据
    """
    # 将数据转换为元组，方便使用 count() 函数统计频率
    data = tuple(data)

    # 找到元组中出现次数最多的元素
    most_common = max(set(data), key=data.count)

    return most_common



def adaptive_hist_eq(image):
    # 将图像转换为LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    # 对亮度通道（L通道）应用自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    # 将图像转换回RGB格式
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result



def image_enhence(image1,image2,image3,image4):
    batches, _, imgH, imgW = image1.size()
    y = torch.zeros(1, 3, imgH, imgW)
    imageheight = int(imgH / 2)
    imagewidth = int(imgW / 2)
    image1 = torch.nn.functional.interpolate(image1, size=(imageheight, imagewidth), mode='bilinear',align_corners=True)
    image2 = torch.nn.functional.interpolate(image2, size=(imageheight, imagewidth), mode='bilinear',align_corners=True)
    image3 = torch.nn.functional.interpolate(image3, size=(imageheight, imagewidth), mode='bilinear',align_corners=True)
    image4 = torch.nn.functional.interpolate(image4, size=(imageheight, imagewidth), mode='bilinear',align_corners=True)

    y[:, :, :imageheight, :imagewidth] = image1
    y[:, :, :imageheight, imagewidth:] = image2
    y[:, :, imageheight:, :imagewidth] = image3
    y[:, :, imageheight:, imagewidth:] = image4
    image=y
    return image



def reset_parameters(model):
    for name, module in model.named_children():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            reset_parameters(module)


"""
标签平滑
可以把真实标签平滑集成在loss函数里面，然后计算loss
也可以直接在loss函数外面执行标签平滑，然后计算散度loss
"""

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑Loss
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        """
        :param classes: 类别数目
        :param smoothing: 平滑系数
        :param dim: loss计算平均值的维度
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = nn.KLDivLoss()

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        # pred=F.softmax(pred, dim=1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # torch.mean(torch.sum(-true_dist * pred, dim=self.dim))就是按照公式来计算损失
        # loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        # 采用KLDivLoss来计算
        loss = self.loss(pred, true_dist)
        return loss


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        targets = targets.to(torch.long)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss