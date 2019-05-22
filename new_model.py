import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils import spectral_norm
# from cgnl_resnet import SpatialCGNLx

MAX_LENGTH = 40

PAD_token = 0
SOS_token = 1
EOS_token = 2

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, int(output_channels / 4), 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(output_channels / 4), int(output_channels / 4), 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(output_channels / 4))
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(int(output_channels / 4), output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out

class cnn_att(nn.Module):
    # input size is 8*8
    def __init__(self, in_channels, out_channels):
        super(cnn_att, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = self.bn4 = nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 4*4

        self.middle_2r_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        # self.interpolation1 = nn.UpsamplingBilinear2d(size=size)  # 8*8

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )


        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x, size):
        x = self.bn(self.conv(x))
        out_trunk = self.bn(self.conv(x))
        out_mpool1 = self.mpool1(x)
        out_middle_2r_blocks = self.bn(self.conv(out_mpool1))
        #
        # out_interp = self.interpolation1(out_middle_2r_blocks) + out_trunk
        out_interp = F.upsample(out_middle_2r_blocks, size=size, mode='bilinear', align_corners=True) + out_trunk
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out_conv1_1_blocks = self.conv1_1_blocks(out_interp)
        out = (1 + out_conv1_1_blocks) * out_trunk
        out_last = self.bn(self.conv(out))

        return out_last

class CNN(nn.Module):
    def __init__(self, imgH, nc, leakyRelu=False):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()
        self.att = cnn_att(512, 512)
        self.conv0 = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.relu0 = nn.ReLU(64)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv1 = nn.Conv2d(64, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.relu1 = nn.ReLU(256)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU(256)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5 = nn.ReLU(512)
        self.bn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(2, 2), stride=(2, 1), padding=(0, 0))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1), dilation=1, ceil_mode=False)
        self.bn6 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU(1024)

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.attention = SelfAttention(1024)
        # self.cgnl = SpatialCGNLx(256, 256, use_scale=False, groups=8, order=3)


    def forward(self, input):
        # conv features
        # conv2 = self.cnn(input)
        # print(self.cnn)
        conv = self.conv0(input)
        conv = self.relu0(conv)
        conv = self.pool0(conv)
        conv = self.conv1(conv)
        conv = self.relu1(conv)
        conv = self.pool1(conv)

        # conv = self.conv2(conv)
        # conv = self.bn2(conv)
        # conv = self.relu2(conv)

        conv = self.conv3(conv)
        conv = self.relu3(conv)
        conv = self.pool1(conv)
        conv = self.conv4(conv)
        conv = self.bn4(conv)
        conv = self.relu4(conv)
        conv = self.att(conv, conv.size()[2:])
        conv = self.conv5(conv)
        conv = self.bn5(conv)
        conv = self.relu5(conv)
        conv = self.pool3(conv)
        conv = self.conv6(conv)
        conv = self.pool4(conv)
        conv = self.bn6(conv)
        conv = self.relu6(conv)
        conv = self.attention(conv)

        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(0, 2, 1)  # [b, w, c]
        output = conv
        return output

class SelfAttention(nn.Module):

    def __init__(self, d):
        super(SelfAttention, self).__init__()

        assert d % 8 == 0
        self.projections = nn.ModuleList([
            spectral_norm(nn.Conv2d(d, d // 8, 1)),
            spectral_norm(nn.Conv2d(d, d // 8, 1)),
            spectral_norm(nn.Conv2d(d, d, 1))
        ])
        self.gamma = nn.Parameter(torch.zeros(1))  # shape [1]

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, d, h, w].
        Returns:
            a float tensor with shape [b, d, h, w].
        """
        b, d, h, w = x.size()

        q = self.projections[0](x)
        k = self.projections[1](x)
        v = self.projections[2](x)

        q = q.view(b, d // 8, h * w).permute(0, 2, 1)
        k = k.view(b, d // 8, h * w)
        v = v.view(b, d, h * w).permute(0, 2, 1)

        attention = torch.bmm(q, k)  # shape [b, h * w, h * w]
        attention = F.softmax(attention, dim=2)

        out = torch.bmm(attention, v)  # shape [b, h * w, d]
        out = out.permute(0, 2, 1).view(b, d, h, w)
        return x + self.gamma * out

# class CNN(nn.Module):
#     def __init__(self, imgH, nc, leakyRelu=False):
#         super(CNN, self).__init__()
#         assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
#
#         ks = [3, 3, 3, 3, 3, 3, 2]
#         ps = [1, 1, 1, 1, 1, 1, 0]
#         ss = [1, 1, 1, 1, 1, 1, 1]
#         nm = [64, 128, 256, 256, 512, 512, 512]
#
#         cnn = nn.Sequential()
#
#         def convRelu(i, batchNormalization=False):
#             nIn = nc if i == 0 else nm[i - 1]
#             nOut = nm[i]
#             cnn.add_module('conv{0}'.format(i),
#                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
#             if batchNormalization:
#                 cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
#             if leakyRelu:
#                 cnn.add_module('relu{0}'.format(i),
#                                nn.LeakyReLU(0.2, inplace=True))
#             else:
#                 cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
#
#         convRelu(0)
#         cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
#         convRelu(1)
#         cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
#         convRelu(2, True)
#         convRelu(3)
#         cnn.add_module('pooling{0}'.format(2),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
#         convRelu(4, True)
#         convRelu(5)
#         cnn.add_module('pooling{0}'.format(3),
#                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
#         convRelu(6, True)  # 512x1x16
#
#         self.cnn = cnn
#
#
#     def forward(self, input):
#         # conv features
#         conv = self.cnn(input)
#         b, c, h, w = conv.size()
#         assert h == 1, "the height of conv must be 1"
#         conv = conv.squeeze(2) # b *512 * width
#         conv = conv.permute(0, 2, 1)  # [b, w, c]
#         output = conv
#         return output


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=self.n_layers, bidirectional=True, dropout=(0 if n_layers == 1 else dropout))

    def forward(self, input, hidden=None):
        self.gru.flatten_parameters()
        outputs, hidden = self.gru(input, hidden)

        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden



# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step).view(1,-1,self.hidden_size)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        self.gru.flatten_parameters()
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.log_softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class model(nn.Module):
    def __init__(self, cnn, encoder, decoder):
        super(model, self).__init__()

        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image, max_length):

        batch_size = image.size()[0]

        input_tensor = self.cnn(image)
        input_tensor = input_tensor.permute(1, 0, 2)

        encoder_outputs, encoder_hidden = self.encoder(
            input_tensor)

        decoder_input = torch.tensor([[SOS_token] * batch_size]).cuda()
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        decoder_outputs = []
        for di in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            decoder_outputs.append(decoder_output)
            # loss += self.criterion(decoder_output, target_tensor[di].squeeze(1))
        decoder_outputs = torch.stack(decoder_outputs, 0)
        return decoder_outputs.permute(1, 0, 2)




