import torch
import numpy
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from gn import GroupNorm

class Atrous_ResNet_features(nn.Module):
    def __init__(self, pretrained=True):
        super(Atrous_ResNet_features, self).__init__()
        resnet = models.resnet152()
        res152_path = '/home/kawhi/.torch/models/resnet152-b121ed2d.pth'
        if pretrained:
            resnet.load_state_dict(torch.load(res152_path))
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n or 'downsample.0' in n:
        #         m.stride = (1, 1)
        # for n, m in self.layer4.named_modules():
        #     if 'conv2' in n or 'downsample.0' in n:
        #         m.stride = (1, 1)
        layer3_group_config = [1, 2, 5, 9]
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
            self.layer3[idx].conv2.padding = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
        layer4_group_config = [5, 9, 17]
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = (layer4_group_config[idx], layer4_group_config[idx])
            self.layer4[idx].conv2.padding = (layer4_group_config[idx], layer4_group_config[idx])

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

class extension_1(nn.Module):
    def __init__(self, num_classes):
        super(extension_1, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet152(pretrained=True)

        self.feature_fuse = fusion(256)
        self.conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1,
                              dilation=1, padding=1, bias=False)
        self.conv1 = resnet.conv1
        self.conv2 = nn.Conv2d(3, 256, kernel_size=3,
                                            stride=1, padding=1, dilation=1)

        self.conv3 = nn.Conv2d(2048, 256, kernel_size=1,
                               stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(1024, 256, kernel_size=1,
                               stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=1,
                               stride=1, padding=0, dilation=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=1,
                               stride=1, padding=0, dilation=1)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=5, dilation=5)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=7, dilation=7)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=11, dilation=11)
        self.conv11= nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=13, dilation=13)

        self.conv12 = nn.Conv2d(768, 256, kernel_size=1,
                               stride=1, padding=0, dilation=1)

        self.conv_end = nn.Conv2d(256, 21, kernel_size=1,
                               stride=1, padding=0, dilation=1)
        self.fc2 = nn.Linear(256, 21)

        # self.bn1 = nn.BatchNorm2d(256)
        self.bn1 = GroupNorm(256)
        self.relu1 = nn.ReLU(inplace=True)

        self.good_resnet = Atrous_ResNet_features(pretrained=True)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(256, 256, 1, stride=1))
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):

        fm1, fm2, fm3, fm4 = self.good_resnet(x)

        level1 = self.conv6(fm1)
        level1 = self.bn1(level1)
        # level1 = self.dense_aspp(fm1)
        level2 = self.conv5(fm2)
        level2 = self.bn1(level2)
        # level2 = self.dense_aspp(fm2)
        fm3 = self.conv4(fm3)
        fm3 = self.bn1(fm3)
        level3 = self.dense_aspp(fm3)
        fm4 = self.conv3(fm4)
        fm4 = self.bn1(fm4)
        level4 = self.dense_aspp(fm4)

        med1 = self.feature_fuse(level2, level1)
        med2 = self.feature_fuse(level3, level2)
        med3 = self.feature_fuse(level4, level3)

        med4 = self.feature_fuse(med2, med1)
        med5 = self.feature_fuse(med3, med2)

        med6 = self.feature_fuse(med5, med4)

        med6 = self.conv6(med6) + fm1
        #
        output = self.conv_end(med6)
        output = F.upsample(output, scale_factor=4, mode='bilinear', align_corners=True)

        return output

    def dense_aspp(self, input):
        dual = self.conv6(input)
        feature_list = []
        # feature_list.append(self.conv7(input))
        # feature_list.append(self.conv8(input))
        # feature_list.append(self.conv9(input))
        feature_list.append(self.conv9(input))
        feature_list.append(self.conv10(input))
        feature_list.append(self.conv11(input))
        output_list = []
        for i in range(len(feature_list)):
            s = feature_list[i]
            for j in range(i):
                s += feature_list[i]
            output_list.append(s)
        dense_output = torch.cat(output_list, 1)
        dense_output = self.conv12(dense_output)

        dense_output = self.relu1(dense_output)
        dense_output = self.bn1(dense_output)
        dense_output = dual + dense_output

        return dense_output

class fusion(nn.Module):
    def __init__(self, in_channel):
        super(fusion, self).__init__()
        # self.bn = nn.BatchNorm2d(in_channel)
        self.bn = GroupNorm(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, dilation=1)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channel, in_channel, 1, stride=1))

    def forward(self, low, high):
        low_down = F.upsample(low, scale_factor=2, mode='bilinear', align_corners=True)
        low_down = self.conv1(low_down)
        low_down = self.bn(low_down)
        low = self.conv1(low)

        high_up = self.conv2(high)
        high_up = self.bn(high_up)
        high =self.conv1(high)
        low = low + high_up
        high = low_down + high
        high = self.conv1(high)
        low = self.global_avg_pool(low)
        output = low * high
        return output

