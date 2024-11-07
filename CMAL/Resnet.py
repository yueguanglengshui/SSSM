import torch.nn as nn
from  torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 这是一个用于图像分类的深度学习模型 ResNet，它包含多个残差块（residual block），每个残差块内部有多个卷积层和批归一化层，其中包括两种不同类型的残差块，
# 即基本残差块（BasicBlock）和瓶颈残差块（Bottleneck）。模型的输入是一张彩色图像，输出是一个分类结果，同时还可以返回多个中间层的输出，
# 这些中间层的输出可以用于一些特定任务，例如物体检测或者分割。在模型的训练过程中，使用了一些技巧来优化模型性能，例如使用了批量归一化、残差连接等。
class ResNet(nn.Module):
    # 这段代码定义了一个ResNet模型类，包括了初始化函数__init__和前向传播函数forward。在初始化函数中，
    # 根据传入的参数配置，创建了一个ResNet模型的各个组成部分，包括卷积层、批归一化层、激活函数、池化层、多个残差块层、
    # 全局平均池化层、全连接层和分类层等。其中，根据传入的block类型、layers层数、num_classes分类数等参数，
    # 使用_make_layer函数创建了多个残差块层。在每个残差块层中，使用了卷积层、批归一化层和激活函数等子层来构建每个残差块，
    # 从而组成一个完整的残差块层。在创建各个组成部分的过程中，使用了预定义的初始化方式对每个子层的权重进行初始化。在前向传播函数中，
    # 根据创建的模型组成部分，对输入的数据进行了前向传播，最终输出了模型的预测结果，以及各个残差块层的输出结果。
    # 该方法接受四个参数：block表示基本的卷积块，planes表示输出通道数，blocks表示卷积块的数量，stride表示卷积的步长，默认值为1，dilate表示是否采用扩张卷积，默认值为False。
    # ResNet(inplanes, planes, **kwargs)
    # _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.classfier = nn.Linear(num_classes, 12)

        # self.insert1 = nn.functional.interpolate(scale_factor=2)  # 2倍插值,效果是图片像素放大为原来的2倍
        # self.insert2 = nn.functional.interpolate(scale_factor=2)  # 2倍插值,效果是图片像素放大为原来的2倍

        # 定义卷积操作
        self.conv_upsampled_x5_x4 = nn.Conv2d(2048, 1024, kernel_size=1)
        # 定义卷积操作
        self.conv_upsampled_x4_x3 = nn.Conv2d(1024, 512, kernel_size=1)

        # 定义下采样卷积操作
        self.conv_subsampled_x3_x4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 输出通道数翻倍为 0124
        self.conv_subsampled_x4_x5= nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)  # 输出通道数翻倍为 0124

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 这是一个在PyTorch框架下的深度神经网络模型中的一个方法（函数），用于构建模型中的一个卷积层。
    #
    # 该方法接受四个参数：block表示基本的卷积块，planes表示输出通道数，blocks表示卷积块的数量，stride表示卷积的步长，默认值为1，dilate表示是否采用扩张卷积，默认值为False。
    #
    # 在方法内部，首先根据传入的参数，确定卷积层中使用的归一化方法(norm_layer)
    # 和下采样(downsample)
    # 方法。如果采用扩张卷积，则将卷积层的扩张率(dilation)
    # 更新，步长(stride)
    # 变为1。如果步长不为1或者输入通道数与输出通道数不匹配，则需要使用下采样方法。然后通过一个循环，将指定数量的卷积块添加到卷积层中。
    #
    # 最后返回一个nn.Sequential对象，其中包含指定数量的卷积块。
    # self.layer1 = self._make_layer(block, 64, layers[0])
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # 这段代码实现了一个深度残差网络（ResNet）的前向传播过程。输入数据; x; 经过卷积、批量归一化、ReLU; 激活函数和最大池化层的处理后得到; x1，然后; x1; 作为输入传递给四个残差块（layer1; ~layer4），得到; x2; 到
    # x5。接下来，x5; 经过平均池化层和全连接层，得到一个特征向量; x，再经过一个线性分类器（classfier），最终得到模型的输出。除了模型的输出外，函数还返回了; x1; 到; x5
    # 这五个特征图，用于可视化和特征提取等应用。
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # # 上采样
        # x5_upsampled = nn.functional.interpolate(x5,scale_factor=2) #self.insert2(x5)
        # # 改变通道数
        # x5_upsampled_x4 = self.conv_upsampled_x5_x4(x5_upsampled)
        # # 进行特征融合
        # # merged_features_x4 = torch.cat([x4, x5_upsampled_x4], dim=1)
        # merged_features_x4 = x4+x5_upsampled_x4*0.2
        # # 上采样
        # x4_upsampled = nn.functional.interpolate(merged_features_x4,scale_factor=2)# self.insert1(merged_features_x4)
        # # 改变通道数
        # x4_upsampled_x3 = self.conv_upsampled_x4_x3(x4_upsampled)
        # # 进行特征融合
        #
        # merged_features_x3 = x3+x4_upsampled_x3*0.2
        #
        # x3_downsampled_x4 = self.conv_subsampled_x3_x4(merged_features_x3)
        # x3_concat_x4 = x3_downsampled_x4*0.2+merged_features_x4
        #
        # x4_downsampled_x5 = self.conv_subsampled_x4_x5(x3_concat_x4)
        # x4_concat_x5 = x4_downsampled_x5*0.2+ x5

        # merged_features_x3 = x3
        #
        # x3_downsampled_x4 = self.conv_subsampled_x3_x4(merged_features_x3)
        # x3_concat_x4 = x4 * 0.7 + x3_downsampled_x4*0.3
        #
        # x4_downsampled_x5 = self.conv_subsampled_x4_x5(x4)
        # x4_concat_x5 =x5* 0.7+x4_downsampled_x5*0.3

        x = self.avgpool(x5)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.classfier(x)
        # return x, x1, x2, x3, x4, x5
        return x2,x3, x4, x5,x
        # return x


# 该方法接受四个参数：block表示基本的卷积块，planes表示输出通道数，blocks表示卷积块的数量，stride表示卷积的步长，默认值为1，dilate表示是否采用扩张卷积，默认值为False。
# _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,**kwargs)
def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        state_dict = torch.load('./checkpoint/resnet50.pth')
        model.load_state_dict(state_dict,strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # 该方法接受四个参数：block表示基本的卷积块，planes表示输出通道数，blocks表示卷积块的数量，stride表示卷积的步长，默认值为1，dilate表示是否采用扩张卷积，默认值为False。
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained=False, progress=True, **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained=False, progress=True, **kwargs)
