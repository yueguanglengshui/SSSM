from __future__ import print_function
import os
from PIL import Image
import torchvision.models
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.mixture import GaussianMixture
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from efficientnetv2_model import efficientnetv2_s as create_model
from utils import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# from transformer import MultiHeadAttention

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
class RFB_modified(nn.Module):  # Receptive Field Block
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu =nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        # self.branch4 = nn.Sequential(
        #     BasicConv2d(in_channel, out_channel, 1),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(1, 11), padding=(0, 5)),
        #     BasicConv2d(out_channel, out_channel, kernel_size=(11, 1), padding=(5, 0)),
        #     BasicConv2d(out_channel, out_channel, 3, padding=11, dilation=11)
        # )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)  # 对连接后的结果卷积
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)  # 直连层shortcut

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # x4 = self.branch4(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))  # 按通道数连接四个分支后1*1卷积

        x = self.relu(x_cat + self.conv_res(x))
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)
        max_ = self.max_pool(x)
        avg = self.fc2(self.relu(self.fc1(avg)))
        max_ = self.fc2(self.relu(self.fc1(max_)))
        x = avg + max_
        return self.sigmoid(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg, max_], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        # qk_scale：指定缩放因子，
        # attn_drop：注意力得分的dropout概率；
        # proj_drop：注意力计算后的投影层的dropout概率。
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww      局部注意力窗口不用管
        self.num_heads = num_heads
        head_dim = dim // num_heads         #每个注意力头的特征维度，即输入特征维度除以头数；好像没用到
        self.scale = qk_scale or head_dim ** -0.5    #缩放因子，用于调整注意力得分的尺度；
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  #dim只有一维，输入时就是1维，可以看着一张图被拉成1维
        # #用于计算查询、键和值的线性映射，将输入张量的特征维度映射到三倍的特征维度上，分别用于计算查询、键和值；
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)   #将注意力计算后的值向量进行线性映射的模块；
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # 将输入张量x通过线性映射self.qkv，映射为形状为(batch_size, num_patches, 3 * num_heads, head_dim)的张量qkv；
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 将qkv张量进行形状变换，变为(3, batch_size, num_heads, num_patches, head_dim)的张量qkv；
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # 将 qkv张量沿着第一个维度分割为三个张量q、k和v；
        q = q * self.scale
        # 对q和k进行缩放和点积操作，得到未归一化的注意力得分张量attn；
        attn = (q @ k.transpose(-2, -1))
        # 如果提供了掩码mask，则在计算注意力得分前，先将掩码与attn进行相加；
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




   # self.att1 = SE(patch=16, in_c=feature_size, num_heads=8)
class SE(nn.Module):
    def __init__(self, patch=8, in_c=512, num_heads=8):   #512通道
        super(SE, self).__init__()
        self.patch = patch
        self.avg = nn.AdaptiveAvgPool2d(patch)   #长宽浓缩变为16*16    通道还是不变维持原来数量
        self.att = Attention(dim=in_c, num_heads=num_heads)   #自注意力机制  num_heads 注意力头
        self.avg2 = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(in_c)

    def forward(self, x):
        b, c, h, w = x.size()     #b批次   c通道数  h 高度  w宽度
        if h > self.patch:          #如果输入的图像还是比较大，就在压缩一边
            x = self.avg(x)
        # x = x.view(b, c, -1).transpose(1, 2)   # 转换维度方便注意力计算
        x = x.reshape(b, c, -1).contiguous().transpose(1, 2)
        x = self.ln(x)    #正则化
        x = self.att(x)   #提取全局特征表示
        x = self.avg2(x.transpose(1, 2))
        out = x.flatten(1)    # 最后一个维度为1，就直接减少减少一个维度
        return out


class BAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(BAM, self).__init__()
        self.channels = channels

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(channels*2, channels//reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels//reduction_ratio, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)
        out = torch.cat((max_out, avg_out), dim=1)
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        out = out.view(out.size(0), out.size(1), 1, 1)

        # 应用注意力权重
        out = out * x

        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-5,
                             momentum=0.01, affine=True)
        self.relu =nn.ReLU()
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_channels += 2
        if with_r:
            in_channels += 1
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels= out_channels,kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])


    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3


def compute_fisher_vector(features):
    num_samples, num_channels, height, width = features.shape
    num_features = num_channels * height * width

    # Reshape features tensor into a 2D matrix
    features_2d = features.view(num_samples, num_features)

    # Compute the GMM parameters
    gmm_model = GaussianMixture(n_components=64)
    gmm_model.fit(features_2d.detach().cpu().numpy())

    # Compute the GMM parameters on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means = torch.Tensor(gmm_model.means_).to(device)
    covariances = torch.Tensor(gmm_model.covariances_).to(device)
    weights = torch.Tensor(gmm_model.weights_).to(device)

    # Compute Fisher Vector
    d_means = features_2d - means
    d_covariances = torch.stack([torch.inverse(cov) for cov in covariances])
    normalized_d_means = d_means * torch.sqrt(weights.unsqueeze(1)) / torch.sqrt(covariances)
    normalized_d_covariances = -0.5 * d_covariances + 0.5 * torch.eye(num_features).to(device)
    fv = torch.cat([normalized_d_means.flatten(), normalized_d_covariances.flatten()], dim=0)

    # Perform power normalization and L2 normalization
    fv = fv.sign() * torch.sqrt(fv.abs())
    fv /= fv.norm()

    # Move the feature vector back to CPU
    fv = fv.cpu().numpy()

    return fv

class Spatialattention(nn.Module):
    def __init__(self, in_channels):
        super(Spatialattention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        attention_map = self.conv(x)  # 进行卷积操作得到注意力图
        attention_weights = F.softmax(attention_map, dim=-1)  # 在最后一个维度上进行 softmax 得到注意力权重
        attention_features = x * attention_weights  # 应用注意力权重
        attention_output = torch.sum(attention_features, dim=1, keepdim=True)  # 在通道维度上求和，保留特征和位置信息
        return attention_output

class Network_Wrapper(nn.Module):
    def __init__(self, model,num_class,K,crop_image_num,partnet=None):
        super().__init__()
        self.Features = model
        # self.partmodel=partnet
        self.num_class = num_class
        self.Lstm =256
        self.K=K
        self.crop_image_num=crop_image_num
        # self.max_pool1 = nn.MaxPool2d(kernel_size=56, stride=1)
        # self.max_pool2 = nn.MaxPool2d(kernel_size=28, stride=1)
        # self.max_pool3 = nn.MaxPool2d(kernel_size=14, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=20, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=20, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=20, stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=10, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        # self.pooling_layer = nn.AdaptiveAvgPool2d((22, 22))
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, ceil_mode=False)
        self.cbam=CBAM(in_channels=512+64)
        # 假设x是输入的特征图张量
        self.ConvTranspose = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)


        self.fc_part1 = nn.Linear(256, 1)
        self.fc_part2 = nn.Linear(256, 1)
        self.fc_part3 = nn.Linear(256, 1)
        self.fc = nn.Linear(256, num_class)

        self.conv_block1 = nn.Sequential(
            RFB_modified(in_channel=128+256, out_channel=512),
            CBAM(in_channels=512),
            # nn.ELU(),

        )
        self.block = nn.Sequential(
            nn.GroupNorm(16, 512+64),
            # nn.BatchNorm2d(512+64),
            CoordConv(in_channels=512+64, out_channels=256),
            # nn.GroupNorm(16, 256),
            # nn.BatchNorm2d(256),
            # nn.ELU(inplace=True),
            # CBAM(in_channels=256),
        )
        #其目的对box进行改进
        self.conv_part_block1 = nn.Sequential(
            # nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            # NonLocalBlock(in_channels=256),
            # # CBAM(in_channels=256),
            # nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            # RFB_modified(in_channel=256, out_channel=100),
            #
            # CBAM(in_channels=100),

            nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            NonLocalBlock(in_channels=256),
            CBAM(in_channels=256),
            RFB_modified(in_channel=256, out_channel=100),
            nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            CBAM(in_channels=100),

            # # CoordConv(in_channels=256, out_channels=128),
            # # # nn.GroupNorm(16, 128),
            # nn.BatchNorm2d(256),
            # # # nn.ELU(inplace=True),
            # # # CBAM(in_channels=128),
            # RFB_modified(in_channel=256, out_channel=100),
            # # nn.GroupNorm(16, 128),
            # nn.BatchNorm2d(100),
            # CBAM(in_channels=100),
        )
        self.classifier_part1 = nn.Sequential(
            nn.LayerNorm(256),
            # nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_class),
        )

        # self.conv_part_transformer1 = nn.Sequential(
        #     BasicConv(256, 128, kernel_size=3, stride=1, padding=1, relu=True),  # [512,256]
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     # RFB_modified(in_channel=256, out_channel=64),
        #     CoordConv(in_channels=128, out_channels=128),
        #     # CBAM(in_channels=256),
        #     # nn.BatchNorm2d(64)
        # )
        self.classifier1 = nn.Sequential(        #SE [batchsize,channels]
            nn.BatchNorm1d(512),
            # nn.LayerNorm(512),
            # nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(256),
            nn.BatchNorm1d(256),
            # nn.ELU(inplace=True),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(256, num_class)
        )

        self.conv_block2 = nn.Sequential(
            RFB_modified(in_channel=160, out_channel=512),
            CBAM(in_channels=512),
            # nn.ELU(),

            # nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            # NonLocalBlock(in_channels=160 + 256),
            # CBAM(in_channels=160 + 256),
            # RFB_modified(in_channel=160 + 256, out_channel=512),
            # nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            # CBAM(in_channels=512)
        )
        self.conv_part_block2 = nn.Sequential(
            # # CoordConv(in_channels=256, out_channels=128),
            # # # nn.GroupNorm(16, 128),
            # nn.BatchNorm2d(256),
            # # # nn.ELU(inplace=True),
            # # # CBAM(in_channels=128),
            # RFB_modified(in_channel=256, out_channel=100),
            # # nn.GroupNorm(16, 128),
            # nn.BatchNorm2d(100),
            # CBAM(in_channels=100),

            nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            NonLocalBlock(in_channels=256),
            CBAM(in_channels=256),
            RFB_modified(in_channel=256, out_channel=100),
            nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            CBAM(in_channels=100),
        )


        self.classifier_part2 = nn.Sequential(
            nn.LayerNorm(256),
            # nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_class),
        )
        # self.conv_part_transformer2 = nn.Sequential(
        #     BasicConv(256, 128, kernel_size=3, stride=1, padding=1, relu=True),  # [512,256]
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     # RFB_modified(in_channel=256, out_channel=64),
        #     CoordConv(in_channels=128, out_channels=128),
        #     # CBAM(in_channels=256),
        #     # nn.BatchNorm2d(64)
        # )
        self.classifier2 = nn.Sequential(
            # nn.BatchNorm1d(256),
            nn.BatchNorm1d(512),
            # nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            # nn.BatchNorm1d(256),
            nn.BatchNorm1d(256),
            # nn.ELU(inplace=True),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(256, num_class),
        )

        self.conv_block3 = nn.Sequential(
            RFB_modified(in_channel=256, out_channel=512),
            CBAM(in_channels=512),
            # nn.ELU(),

            # nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            # NonLocalBlock(in_channels=256),
            # CBAM(in_channels=256),
            # RFB_modified(in_channel=256, out_channel=512),
            # nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            # CBAM(in_channels=512)
        )
        self.conv_part_block3 = nn.Sequential(
            # # CoordConv(in_channels=256, out_channels=128),
            # # # nn.GroupNorm(16, 128),
            # nn.BatchNorm2d(256),
            # # # nn.ELU(inplace=True),
            # # # CBAM(in_channels=128),
            # RFB_modified(in_channel=256, out_channel=100),
            # # nn.GroupNorm(16, 128),
            # nn.BatchNorm2d(100),
            # CBAM(in_channels=100),

            nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            NonLocalBlock(in_channels=256),
            CBAM(in_channels=256),
            RFB_modified(in_channel=256, out_channel=100),
            nn.LocalResponseNorm(size=3, alpha=1e-6, beta=0.75, k=1.0),
            CBAM(in_channels=100),
        )
        self.classifier_part3 = nn.Sequential(
            nn.LayerNorm(256),
            # nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_class),
            # nn.Flatten(),
            # nn.LayerNorm(128),
            # nn.ELU(inplace=True),
            # nn.Linear(128, 512),  # 8*8*64
            # nn.ELU(inplace=True),
            # nn.LayerNorm(512),
            # nn.Dropout(p=0.2),
            # nn.Linear(1024, self.Lstm), #256这表示特征长度用于LSTM
        )

        # self.conv_part_transformer3 = nn.Sequential(
        #     BasicConv(256, 128, kernel_size=3, stride=1, padding=1, relu=True),  # [512,256]
        #     # nn.AvgPool2d(kernel_size=2, stride=2),
        #     # RFB_modified(in_channel=256, out_channel=64),
        #     CoordConv(in_channels=128, out_channels=128),
        #     # CBAM(in_channels=256),
        #     # nn.BatchNorm2d(64)
        # )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            # nn.ELU(inplace=True),
            nn.Linear(256, num_class),
        )

        self.classifier_concat = nn.Sequential(
            # nn.BatchNorm1d(256 * 3),
            nn.LayerNorm(512*3),
            nn.Dropout(p=0.1),
            nn.Linear(512 * 3, 512),
            # nn.BatchNorm1d(512),
            nn.LayerNorm(512),
            # nn.ELU(inplace=True),
            nn.ReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_class),
        )
        # self.spatial_attention = Spatialattention(256)
        # self.RNNpartconcat = RNNpart(121, num_class)
        self.RNNpartLSTM1 =RNNpart(100, num_class)   #121和256表示特征长度
        self.RNNpartLSTM2 = RNNpart(100, num_class)
        self.RNNpartLSTM3 = RNNpart(100, num_class)
        # self.RNNpartLstmConcat = RNNpartcat(128, num_class)  # 121和256表示特征长度
        self.RNNLSTM1 = RNNLSTM(400,num_class)
        self.RNNLSTM2 = RNNLSTM(400, num_class)
        self.RNNLSTM3 = RNNLSTM(400, num_class)      #484=22*22
        # self.RNNLSTM4 = RNNLSTM(484, num_class)

        self.partdecoder1 = partATTmodel1(num_class=num_class, channal=512)
        self.partdecoder2 = partATTmodel2(num_class=num_class, channal=512)
        self.partdecoder3 = partATTmodel3(num_class=num_class, channal=512)
        # self.partATT =  partATTmodel(num_class=num_class, channal=512)

    def forward(self, x,index=None,K=None,crop_image_num=None,weight_loss=None,isBranch=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crop_image_total = K * crop_image_num
        # if isBranch == False:
        map_branch,x1, x2, x3,classifier= self.Features(x)
        map_branch = self.pooling_layer(map_branch)
        #将特征图大小都变为统一大小22
        x3=self.ConvTranspose(x3)
        # 12, 128, 16, 16
        # 12, 160, 16, 16
        # 12, 256, 8, 8
        # x2 = torch.cat((x2, x3), dim=1)   # 12, 160+256, 16, 16
        # x1 = torch.cat((x1, x3), dim=1) # 12, 128+256, 16, 16
        # x3 = torch.cat((x3, x2), dim=1)  # 12, 160+256, 16, 16
        # x2 = torch.cat((x2, x1), dim=1)  # 12, 160+128, 16, 16
        # x2 = torch.cat((x2, x3), dim=1)  # 12, 160+256, 16, 16
        # x1 = torch.cat((x1, x3), dim=1)  # 12, 128+256, 16, 16

        # x1 = torch.cat((x1, map_branch), dim=1)
        # x2 = torch.cat((x2, map_branch), dim=1)
        # x3 = torch.cat((x3, map_branch), dim=1)
        # x2 = torch.cat((x2, x3), dim=1)   # 12, 160+256, 16, 16
        x1 = torch.cat((x1, x3), dim=1) # 12, 128+256, 16, 16
        #向下
        x1_ = self.conv_block1(x1)
        part1 = torch.cat((x1_, map_branch), dim=1)
        part1=self.cbam(part1)
        # part1 = self.cbam(part1)
        x1_part = self.block(part1)
        x1_part_box = self.block(part1)
        # x1_part = torch.cat((x1_part, map_branch), dim=1)
        # x1_part=self.cbam(x1_part)
        # x2=torch.cat((x1_part, x2), dim=1)

        x2_ = self.conv_block2(x2)
        part2 = torch.cat((x2_, map_branch), dim=1)
        part2 = self.cbam(part2)
        x2_part = self.block(part2)
        x2_part_box = self.block(part2)
        # x2_part = torch.cat((x2_part, map_branch), dim=1)
        # x2_part = self.cbam(x2_part)
        # x2_part_scale = self.pool(x2_part)
        # x3 = torch.cat((x2_part, x3), dim=1)

        x3_ = self.conv_block3(x3)
        part3 = torch.cat((x3_, map_branch), dim=1)
        part3 = self.cbam(part3)
        x3_part = self.block(part3)
        x3_part_box = self.block(part3)
        # x3_part = torch.cat((x3_part, map_branch), dim=1)
        # x3_part = self.cbam(x3_part)

        # featuremap_part1 = x1_part
        # featuremap_part2 = x2_part
        # featuremap_part3 = x3_part


        #向上
        # x3_ = self.conv_block3(x3)
        # x3_part = self.block(x3_)
        #
        # x2 = torch.cat((x3_part, x2), dim=1)
        # x2_ = self.conv_block2(x2)
        # x2_part = self.block(x2_)
        #
        # x1 = torch.cat((x2_part, x1), dim=1)
        # x1_ = self.conv_block1(x1)
        # x1_part = self.block(x1_)


        featuremap1 = x1_.detach()
        # x1_=torch.cat((x1_, featuremap_part1), dim=1)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)



        featuremap2 = x2_.detach()
        # x2_ = torch.cat((x2_, featuremap_part2), dim=1)
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)


        featuremap3 = x3_.detach()
        # x3_ = torch.cat((x3_, featuremap_part3), dim=1)
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)


        # 获取三个分支的特征信息，构建新的特征图
        if (index==0):
            p1 = self.state_dict()['classifier1.1.weight']
            p2 = self.state_dict()['classifier1.5.weight']
            att_map_1 = map_generate(featuremap1, x3_c, p1, p2)
            # inputs1_att, boxnum,cropped_images_points1 = attention_im(x, att_map_1, K, crop_image_num, index=1,branch=1)
            p1 = self.state_dict()['classifier2.1.weight']
            p2 = self.state_dict()['classifier2.5.weight']
            att_map_2 = map_generate(featuremap2, x2_c, p1, p2)
            p1 = self.state_dict()['classifier3.1.weight']
            p2 = self.state_dict()['classifier3.5.weight']
            att_map_3 = map_generate(featuremap3, x1_c, p1, p2)
            inputs_ATT = highlight(x, att_map_1, att_map_2, att_map_3,K)
            return x1_c, x2_c, x3_c, x_c_all, inputs_ATT,
        if (index ==1 or index==2 or index==3):
            # 第一分支
            boxnum1, cropped_images_points1 = attention1(featuremap1, K,crop_image_num, index=index, branch=1)
            # inputs_ATT1, boxnum1, cropped_images_points1 = attention_im(x, att_map_1.squeeze(), K, crop_image_num,index=index, branch=1)
            batches1, channels1, imgH1, imgW1 = x1_part_box.size()
            feature_size_H1 = int(imgH1 /2)  #10
            feature_size_W1 = int(imgW1 /2)
            batch_box_classifier1=torch.empty((batches1, self.num_class)).cuda()
            #第二分支
            boxnum2, cropped_images_points2 = attention2(featuremap2, K, crop_image_num, index=index, branch=2)
            # inputs_ATT2, boxnum2, cropped_images_points2 = attention_im(x, att_map_2.squeeze(), K,crop_image_num, index=index, branch=2)
            batches2, channels2, imgH2, imgW2 = x2_part_box.size()
            feature_size_H2 = int(imgH2 / 2)
            feature_size_W2 = int(imgW2 / 2)
            batch_box_classifier2 = torch.empty((batches2, self.num_class)).cuda()
            #第三分支
            # inputs_ATT3, boxnum3,cropped_images_points3 = attention_im(x,  att_map_3.squeeze(), K, crop_image_num, index=index,branch=3)
            boxnum3, cropped_images_points3 = attention3(featuremap3, K, crop_image_num, index=index, branch=3)
            batches3, channels3, imgH3, imgW3 = x3_part_box.size()
            feature_size_H3 = int(imgH3/2)
            feature_size_W3 = int(imgW3/2)
            batch_box_classifier3 = torch.empty((batches3, self.num_class)).cuda()

            batch_box_classifier = torch.empty((batches3, self.num_class)).cuda()
            for batch_index in range(batches1):
                image1 = x1_part_box[batch_index]
                # 创建一个大小为 (batch_size, image_height, image_width) 的空张量
                batch_box_image1 = torch.empty((boxnum1[batch_index], channels1,feature_size_H1, feature_size_W1))

                for i in range(boxnum1[batch_index]):
                    x_min, y_min, x_max, y_max = cropped_images_points1[batch_index][i]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)  # 将浮点数转换为整数
                    cropped_image1 = image1[:, x_min:x_max, y_min:y_max]
                    # # 使用PyTorch的F.interpolate函数将裁切的图像调整到指定大小
                    transformed_image1 = F.interpolate(
                        cropped_image1.unsqueeze(0),
                        size=feature_size_H1,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)
                    batch_box_image1[i]= transformed_image1  # [boxnum[batch_index], channels1,feature_size_H1, feature_size_W1]

                image2 = x2_part_box[batch_index]
                # 创建一个大小为 (batch_size, image_height, image_width) 的空张量
                batch_box_image2 = torch.empty((boxnum2[batch_index], channels2, feature_size_H2, feature_size_W2))
                for i in range(boxnum2[batch_index]):
                    x_min, y_min, x_max, y_max = cropped_images_points2[batch_index][i]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)  # 将浮点数转换为整数
                    cropped_image2 = image2[:, x_min:x_max, y_min:y_max]
                    # # 使用PyTorch的F.interpolate函数将裁切的图像调整到指定大小
                    transformed_image2 = F.interpolate(
                        cropped_image2.unsqueeze(0),
                        size=feature_size_H2,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)
                    batch_box_image2[i] = transformed_image2  # [boxnum[batch_index], channels1,feature_size_H1, feature_size_W1]

                image3 = x3_part_box[batch_index]
                # 创建一个大小为 (batch_size, image_height, image_width) 的空张量
                batch_box_image3 = torch.empty((boxnum3[batch_index], channels3, feature_size_H3, feature_size_W3))
                # batch_box_classifier1=torch.empty((boxnum[batch_index], 256))  #原本256应该是类别数量，但是为了在LSTM中更好的运算，改成了256
                for i in range(boxnum3[batch_index]):
                    x_min, y_min, x_max, y_max = cropped_images_points3[batch_index][i]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)  # 将浮点数转换为整数
                    cropped_image3 = image3[:, x_min:x_max, y_min:y_max]
                    # # 使用PyTorch的F.interpolate函数将裁切的图像调整到指定大小
                    transformed_image3 = F.interpolate(
                        cropped_image3.unsqueeze(0),
                        size=feature_size_H3,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)
                    batch_box_image3[i] = transformed_image3  # [boxnum[batch_index], channels1,feature_size_H1, feature_size_W1]
                # 通过运算


                #通过运算
                #第一分支
                convx1 = batch_box_image1
                convx1 = convx1.cuda()
                convx1 = self.conv_part_block1(convx1)  #[boxnum, 128,11, 11]
                # convx1=self.max_pool(convx1)
                # convx1 = convx1.view(convx1.size(0), -1)  #[boxnum,128]

                convx1 = torch.flatten(convx1, start_dim=2)   #[boxnum, 128,11*11]
                convx1 = torch.reshape(convx1, (-1, convx1.size(-1)))
                convx1 = convx1.unsqueeze(1) #[boxnum,1,128]

                batch_box_classifier1[batch_index] = self.RNNpartLSTM1(convx1).squeeze(0)


                # 第二分支
                convx2 = batch_box_image2
                convx2 = convx2.cuda()
                convx2 = self.conv_part_block2(convx2)  # [boxnum, 64,8, 8]

                # convx2_1 = self.max_pool(convx2)
                # convx2_1 = convx2_1.view(convx2_1.size(0), -1)  # [boxnum,128]

                convx2 = torch.flatten(convx2, start_dim=2)  # [boxnum, 128,11*11]
                convx2 = torch.reshape(convx2, (-1, convx2.size(-1)))

                # convx2 = torch.cat((convx2_1, convx2_2), dim=0)
                convx2 = convx2.unsqueeze(1)
                batch_box_classifier2[batch_index] = self.RNNpartLSTM2(convx2).squeeze(0)


                # 第三分支
                # 通过运算
                convx3 = batch_box_image3
                convx3 = convx3.cuda()
                convx3 = self.conv_part_block3(convx3)  # [boxnum, 64,8, 8]
                convx3 = self.max_pool(convx3)
                convx3 = convx3.view(convx3.size(0), -1)  # [boxnum,128]

                # convx3 = torch.flatten(convx3, start_dim=2)  # [boxnum, 128,11*11]
                # convx3 = torch.reshape(convx3, (-1, convx3.size(-1)))
                convx3 = convx3.unsqueeze(1)
                batch_box_classifier3[batch_index] = self.RNNpartLSTM3(convx3).squeeze(0)

                #将三个分支的convx拼起来
                # convxconcat = torch.cat((convx1, convx2, convx3), dim=0)
                # batch_box_classifier[batch_index] = self.RNNpartLstmConcat(convxconcat).squeeze(0)

            # 第一分支
            #将特征图变小
            # x1_part = self.conv_part_transformer1(x1_part)  # [12,128,11,11]
            #通过transformer解码器
            part1 = self.partdecoder1(x1_part)  # [12,128,22*22]
            #通过一个LSTM变成[12,100]
            part1 = self.RNNLSTM1(part1)
            # part1 = self.classifier_part1(part1)
            # 第二分支
            part2 = self.partdecoder2(x2_part)
            part2 = self.RNNLSTM2(part2)
            # part2 = self.classifier_part2(part2)
            # 第三分支
            part3 = self.partdecoder3(x3_part)  # [128,12,121]
            part3 = self.RNNLSTM3(part3)
            # part3=self.classifier_part3(part3)
            #将三个层次相互融合
            #融合后通过一个tansformer
            #对三个分支进行transformer或者更细致的卷积操作

            # batches1, channels1, imgH1, imgW1 = x1_part.size()
            # x3_part = F.interpolate(
            #     x3_part,
            #     size=(imgH1,imgW1),
            #     mode='bilinear',
            #     align_corners=True
            # )
            # x_part = torch.cat((x1_part, x2_part, x3_part), dim=1) #[12,256*3,22,22]
            # x_part=self.conv_part_transformer(x_part)     #降低特征图大小  #[12,256*3,22,22]
            # part_classifier = self.partATT(x_part)     #[12,256*3,22*22] part_classifier [256*3,12,22*22]
            # part_classifier = self.RNNLSTM4(part_classifier)
            # part_classifier=self.RNNpartconcat(part_classifier)


            # 使用 torch.stack 函数将它们堆叠起来
            # stacked_tensor = torch.cat((part_classifier11, part_classifier22, part_classifier33), dim=1)
            # stacked_tensor = self.RNNpartconcat(stacked_tensor)   #[16,100]
            inputsATT=x
            # inputsATT=(inputs_ATT1+inputs_ATT2+inputs_ATT3)/3
            # part_classifier=batch_box_classifier+batch_box_classifier1+part1+batch_box_classifier2+part2+batch_box_classifier3+part3+part_classifier+x_c_all+x2_c+ x3_c+ x1_c
            # part_classifier=part_classifier/12
            # return x_c_all + part_classifier, batch_box_classifier1 + part1 + x1_c, batch_box_classifier2 + part2 + x2_c, batch_box_classifier3 + part3 + x3_c, inputsATT
            return classifier,batch_box_classifier1+part1+x1_c,batch_box_classifier2+part2+x2_c,batch_box_classifier3+part3+x3_c,inputsATT
class partATTmodel(nn.Module):
    def __init__(self, num_class, channal):
        super().__init__()
        self.num_class = num_class
        self.decoder = LightweightTransformerDecoder(16*16, 16, 16*16, 5)  # [特征长度，注意力头数量，中间节点数量，层数]
        self.classifier_part = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128*11*11, 128*11),
            nn.LayerNorm(128*11),
            nn.Dropout(p=0.2),
            nn.ELU(inplace=True),
            nn.Linear(128*11, num_class),
        )
        self.linear = nn.Linear(22*22, num_class)
    def forward(self, x, index=None, map1=None, map2=None, map3=None, cropped_images_points1=None,cropped_images_points2=None, cropped_images_points3=None):
        # convx3 = self.conv_block_part3(x)  # convx1 [10,256,12,12]
        # 将convx1形状变为[12,128,11*11]的张量
        convx = torch.flatten(x, start_dim=2)  # start_dim表示从第几维开始展平，默认为2
        convx = self.decoder(convx)  # [128,12,11*11]
        # convx = convx.transpose(0, 1) # [12,128,11*11]
        # convx = self.classifier_part(convx)
        # convx = convx[-1]
        # convx=self.linear(convx)
        # self.linear1 = nn.Linear(d_model, d_ff)
        part_classifier = convx
        return part_classifier

class partATTmodel3(nn.Module):
    def __init__(self, num_class, channal):
        super().__init__()
        self.num_class = num_class
        self.decoder3 = LightweightTransformerDecoder(20*20, 10, 20*20, 5)  # [特征长度，注意力头数量，中间节点数量，层数]
        self.linear = nn.Linear(22*22, num_class)
        # self.classifier_part3 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(3872, 1024),
        #     nn.ELU(inplace=True),
        #     nn.Linear(1024, num_class),
        # )
    def forward(self, x, index=None, map1=None, map2=None, map3=None, cropped_images_points1=None,cropped_images_points2=None, cropped_images_points3=None):
        # convx3 = self.conv_block_part3(x)  # convx1 [10,256,12,12]
        # 将convx1形状变为[12,64,11*11]的张量
        convx3 = torch.flatten(x, start_dim=2)  # start_dim表示从第几维开始展平，默认为2
        convx3 = self.decoder3(convx3)  # [12,64,11*11]
        # convx3 = self.classifier_part3(convx3)
        # convx3 = convx3[-1]
        # convx3 = self.linear(convx3)
        part_classifier3 = convx3
        return part_classifier3

class partmodel3(nn.Module):
    def __init__(self, num_class, channal):
        super().__init__()
        self.num_class = num_class
        self.classifier_part3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),  # 8*8*64
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(1024, num_class),
        )
    def forward(self, x, index=None, map1=None, map2=None, map3=None, cropped_images_points1=None,
                cropped_images_points2=None, cropped_images_points3=None):
        # convx3 = self.conv_block_part3(x)  # convx1 [10,256,12,12]
        convx3 = self.classifier_part3(x)
        part_classifier3 = convx3
        return part_classifier3

class partATTmodel2(nn.Module):
    def __init__(self, num_class, channal):
        super().__init__()
        self.num_class = num_class
        self.decoder2 = LightweightTransformerDecoder(20*20, 10, 20*20, 5)  # [特征长度，注意力头数量，中间节点数量，层数]
        # self.classifier_part2 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(3872, 1024),
        #     # nn.LayerNorm(256),
        #     nn.ELU(inplace=True),
        #     nn.Linear(1024, num_class),
        # )

        self.linear = nn.Linear(22 * 22, num_class)

    def forward(self, x, index=None, map1=None, map2=None, map3=None, cropped_images_points1=None,
                cropped_images_points2=None, cropped_images_points3=None):
        # convx2 = self.conv_block_part2(x)  # convx1 [10,256,12,12]
        # 将convx1形状变为[12,64,11*11]的张量
        convx2 = torch.flatten(x, start_dim=2)  # start_dim表示从第几维开始展平，默认为2
        convx2 = self.decoder2(convx2)  # [12,64,11*11]
        # convx2 = convx2[-1]
        # convx2 = self.linear(convx2)
        # convx2 = self.classifier_part2(convx2)
        part_classifier2 = convx2
        return part_classifier2

class partmodel2(nn.Module):
    def __init__(self, num_class, channal):
        super().__init__()
        self.num_class = num_class
        self.classifier_part2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),  # 8*8*64
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(1024, num_class),
        )


    def forward(self, x, index=None, map1=None, map2=None, map3=None, cropped_images_points1=None,
                cropped_images_points2=None, cropped_images_points3=None):
        # convx2 = self.conv_block_part2(x)  # convx1 [10,256,12,12]
        convx2 = self.classifier_part2(x)
        part_classifier2 = convx2
        return part_classifier2

class partATTmodel1(nn.Module):
    def __init__(self, num_class, channal):
        super().__init__()
        self.num_class = num_class
        self.decoder1 = LightweightTransformerDecoder(20*20, 10, 20*20, 5)  # [特征长度，注意力头数量，中间节点数量，层数]
        # self.classifier_part1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(3872, 1024),
        #     # nn.LayerNorm(256),
        #     nn.ELU(inplace=True),
        #     nn.Linear(1024, num_class),
        # )

        self.linear = nn.Linear(22 * 22, num_class)

    def forward(self, x, index=None, map1=None, map2=None, map3=None, cropped_images_points1=None,
                cropped_images_points2=None, cropped_images_points3=None):
        # convx1 = self.conv_block_part1(x)  # convx1 [10,256,12,12]
        # 将convx1形状变为[12,64,11*11]的张量
        convx1 = torch.flatten(x, start_dim=2)  # start_dim表示从第几维开始展平，默认为2
        convx1=self.decoder1(convx1)              #[12,256,11*11]
        # convx1 = convx1[-1]
        # convx1 = self.linear(convx1)
        # convx1 = self.classifier_part1(convx1)
        part_classifier1 = convx1
        return part_classifier1

class partmodel1(nn.Module):
    def __init__(self, num_class, channal):
        super().__init__()
        self.num_class = num_class
        self.classifier_part1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024), #8*8*64
            # nn.LayerNorm(256),
            nn.ELU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(1024, num_class),
        )


    def forward(self, x, index=None, map1=None, map2=None, map3=None, cropped_images_points1=None,
                cropped_images_points2=None, cropped_images_points3=None):
        # convx1 = self.conv_block_part1(x)  # convx1 [12,64,5,5]

        convx1 = self.classifier_part1(x)
        part_classifier1 = convx1
        return part_classifier1





class LightweightTransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        super(LightweightTransformerDecoder, self).__init__()

        self.self_attention = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads) for _ in range(num_layers)]
        )
        self.ffn = nn.ModuleList(
            [TransformerFeedForward(d_model, d_ff) for _ in range(num_layers)]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        for i in range(len(self.self_attention)):
            self_attention_output, _ = self.self_attention[i](x, x, x)
            x = x + self_attention_output
            x = self.layer_norms[i](x)

            ffn_output = self.ffn[i](x)
            x = x + ffn_output
            x = self.layer_norms[i](x)

        return x


class TransformerFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(TransformerFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class RNNpart(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNNpart, self).__init__()
        self.hidden_size = 1024
        # self.attention = nn.MultiheadAttention(input_size, num_heads=11)
        # self.rnn = nn.RNN(input_size,1024, batch_first=False)
        # self.rnn = nn.RNN(input_size, 512, batch_first=False)
        self.lstm = nn.LSTM(input_size, 256, bidirectional=False,batch_first=False)
        self.gru = nn.GRU(input_size, 256, batch_first=False)
        # self.rnn3 = nn.RNN(512, 256, batch_first=False)
        self.fc = nn.Linear(256, output_size)
        self.activation =nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        # out = x.transpose(0, 1)
        # out, attention_weights =self.attention(out,out,out)
        # out, _ = self.rnn(x)
        # out = self.dropout1(out)
        # out, _ = self.rnn2(out)
        out, _ = self.lstm(x)
        # out, _ = self.gru(x)
        # out = torch.mean(out, dim=0)
        out = self.dropout2(out)
        out=self.activation(out[-1,:, :])
        # out = self.dropout2(out)
        # out = self.activation(out)
        out=self.fc(out)

        # out= out[-1]
        # out = self.dropout2(out)
        # out, _ = self.rnn3(out)
        # out = self.fc()  # 仅获取最后一个时间步的输出
        return out
class RNNpartcat(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNNpartcat, self).__init__()
        self.hidden_size = 1024
        # self.attention = nn.MultiheadAttention(input_size, num_heads=11)
        # self.rnn = nn.RNN(input_size,512, batch_first=False)
        # self.rnn2 = nn.RNN(2048, 1024, batch_first=False)
        self.lstm = nn.LSTM(input_size, 256, batch_first=False)
        # self.rnn3 = nn.RNN(512, 256, batch_first=False)
        self.fc = nn.Linear(256, output_size)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        # out = x.transpose(0, 1)
        # out, attention_weights =self.attention(out,out,out)
        # out, _ = self.rnn(x)
        # out = self.dropout1(out)
        # out, _ = self.rnn2(out)
        out, _ = self.lstm(x)
        # out= out[-1]
        # out = self.dropout2(out)
        # out, _ = self.rnn3(out)
        out = self.activation(out[-1, :, :])
        out = self.fc(out)  # 仅获取最后一个时间步的输出
        return out
class RNNLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNNLSTM, self).__init__()
        self.hidden_size = 1024
        # self.attention = nn.MultiheadAttention(input_size, num_heads=10)
        # self.rnn = nn.RNN(input_size,512, batch_first=False)
        # self.rnn2 = nn.RNN(2048, 1024, batch_first=False)
        self.lstm = nn.LSTM(input_size, 256,bidirectional=False,batch_first=False)
        # self.rnn3 = nn.RNN(512, 256, batch_first=False)
        self.fc = nn.Linear(256, output_size)
        self.activation =nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self, x):
        # out = x.transpose(0, 1)
        # out, attention_weights =self.attention(out,out,out)
        # out, _ = self.rnn(out)
        # out = self.dropout1(out)
        # out, _ = self.rnn2(out)
        out, _ = self.lstm(x)

        # out=out[-1, :, :]
        # out, _ = self.rnn3(out)
        # out = torch.mean(out, dim=0)
        out = self.dropout2(out)
        out=self.activation(out[-1,:, :])
        #

        # out = self.activation(out)
        out = self.fc(out)  # 仅获取最后一个时间步的输出
        return out

def fisher_vector_encoding(features, n_components=32):
    """
    Fisher Vector 编码方法实现
    :param features: 特征信息，形状为 (n_samples, n_channels, height, width)
    :param n_components: GMM 模型中的组件数量
    :return: Fisher Vector 编码结果，形状为 (n_samples, n_components * n_channels * 2)
    """
    # 重塑特征
    n_samples, n_channels, height, width = features.shape
    features = features.reshape(n_samples, n_channels * height * width)

    # 训练 GMM 模型
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(features)

    # 计算梯度向量
    grad_vectors = gmm.predict_proba(features)
    means = gmm.means_
    covariances = gmm.covariances_
    inv_covariances = np.linalg.inv(covariances)

    # 进行加权拼接，生成 Fisher Vector 编码
    fisher_vector = np.zeros((n_samples, n_components * n_channels * 2))
    for i in range(n_samples):
        feat = features[i, :].reshape(1, -1)
        prob = grad_vectors[i, :].reshape(-1, 1)
        # 计算 gmm 组件对梯度的贡献
        s1 = np.sqrt(prob) * (feat - means) @ inv_covariances
        s2 = np.sqrt(prob) * (feat - means) @ inv_covariances * (feat - means)
        fisher_vector[i, :] = np.hstack((s1.flatten(), s2.flatten()))

    # 进行 L2 归一化
    fisher_vector = fisher_vector / np.sqrt(np.sum(fisher_vector ** 2, axis=1, keepdims=True))

    return fisher_vector

#图卷积
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, x, adj):
        h = torch.mm(x, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)


class GAT(nn.Module):
    def __init__(self, n_features, n_classes, dropout=0.5, n_heads=1):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attentions = nn.ModuleList([GraphAttentionLayer(n_features, n_features) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(n_features * n_heads, n_classes)

    def forward(self, x, adj):

        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=False, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.sub_sample = sub_sample

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                          padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                               padding=0)

        if sub_sample:
            self.g = nn.Sequential(
                self.g,
                nn.MaxPool2d(kernel_size=(2, 2))
            )
            self.phi = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W(y)
        z = W_y + x

        return z