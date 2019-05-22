import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
import mit_resnet
from gn import GroupNorm

MAX_LENGTH = 50

class CNN_my(nn.Module):
    #                   64    1   37     256
    def __init__(self, imgH, nc, leakyRelu=False):
        super(CNN_my, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

    def forward(self, input):
        x1 = self.pool1(input)
        x2 = self.pool1(x1)
        x3 = self.pool2(x2)
        output = self.pool2(x3)

        return output



class CNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, leakyRelu=False):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # ks = [3, 3, 3, 3, 3, 3, 2]
        # ps = [1, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]
        #
        # cnn = nn.Sequential()
        #
        # def convRelu(i, batchNormalization=False):
        #     nIn = nc if i == 0 else nm[i - 1]
        #     nOut = nm[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        #     if leakyRelu:
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2, inplace=True))
        #     else:
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        #
        # convRelu(0)
        # cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        # convRelu(1)
        # cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        # convRelu(2, True)
        # convRelu(3)
        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        # convRelu(4, True)
        # convRelu(5)
        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        # convRelu(6, True)  # 512x1x16
        #
        # self.cnn = cnn
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.pool2 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        self.orig_resnet = mit_resnet.__dict__['resnet50'](pretrained=True)
        self.net_encoder = ResnetDilated(self.orig_resnet,
                                         dilate_scale=8)
        self.conv = nn.Conv2d(1, 3, kernel_size=3,
                               stride=1, padding=1, dilation=1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.gn1 = GroupNorm(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=(2,1))
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.gn2 = GroupNorm(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=(2,1), padding=(0,1))
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0)


    def forward(self, input):
        asd = input

        input = self.conv1(input)
        input = self.relu(input)
        input = self.pool1(input)
        input = self.conv2(input)
        input = self.pool2(input)
        input = self.conv3(input)
        input = self.gn1(input)
        input = self.conv4(input)
        input = self.pool3(input)
        input = self.conv5(input)
        input = self.gn2(input)
        input = self.conv6(input)
        input = self.pool4(input)
        conv = self.conv7(input)

        # f = self.net_encoder(input)
        # fm1, fm2, fm3, fm4 = f[0], f[1], f[2], f[3]
        # conv2 = self.cnn(asd)
        # print(conv.size()) batch_size*512*1*with
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(0, 2, 1)  # [w, b, c]
        #print(conv.size()) # width batch_size channel
        # rnn features
        output = conv
        #print(output.size(0))
        # print(output.size())# width*batch_size*nclass
        return output

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batchSize):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batchSize = batchSize
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        input = input.permute(1,0,2)
        hidden = hidden.permute(1,0,2)

        self.gru.flatten_parameters()
        output, hidden = self.gru(input, hidden)
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, batchSize, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batchSize = batchSize
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.encoder_out = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        hidden = hidden.permute(1,0,2)

        embedded = self.embedding(input).view(1, self.batchSize, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        encoder_outputs = self.encoder_out(encoder_outputs)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs)

        output = torch.cat((embedded.squeeze(0), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        self.gru.flatten_parameters()
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=True):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]

class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=True):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]