import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models

MAX_LENGTH = 40

PAD_token = 0
SOS_token = 1
EOS_token = 2


class ResidualBlock(nn.Module):
    '''
    
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        resisdual = x if self.right is None else self.right(x)
        out += resisdual
        return F.relu(out)

class ResNet(nn.Module):
    '''
    
    '''

    def __init__(self):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        )

       
        self.layer1 = self._make_layer(128, 128, 1)
        self.layer2 = self._make_layer(256, 256, 2)
        self.layer3 = self._make_layer(512, 512, 5)
        self.layer4 = self._make_layer(512, 512, 3)


        self.pool2 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(128, 256, 3, 1, 1)

        self.pool3 = nn.MaxPool2d(2, 2, 0)
        self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)

        self.pool4 = nn.MaxPool2d(2, (2,1), (1,0))
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5 = nn.Conv2d(512, 512, 2, (2,1), (0,1))
        self.conv6 = nn.Conv2d(512, 512, 2, 2, (0,0))


    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''

        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.pool2(x)
        x = self.layer1(x)
        x = self.conv2(x)

        x = self.pool3(x)
        x = self.layer2(x)
        x = self.conv3(x)

        x = self.pool4(x)
        x = self.layer3(x)
        x = self.conv4(x)

        x = self.layer4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x

class CNN_res(nn.Module):
   
    def __init__(self, imgH):
        super(CNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.resnet = ResNet()


    def forward(self, input):
       
        conv = self.resnet(input)
        
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(0, 2, 1)  # [b, w, c]
        
        output = conv
       
        return output


class CNN(nn.Module):
    def __init__(self, imgH, nc, leakyRelu=False):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

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


    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(0, 2, 1)  # [w, b, c]
        output = conv
        return output


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=self.n_layers, bidirectional=True)

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


######################################################################
# Define Decoder
# --------------
#
# Similarly to the ``EncoderRNN``, we use the ``torch.nn.GRU`` module for
# our decoder’s RNN. This time, however, we use a unidirectional GRU. It
# is important to note that unlike the encoder, we will feed the decoder
# RNN one word at a time. We start by getting the embedding of the current
# word and applying a
# `dropout <https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.Dropout>`__.
# Next, we forward the embedding and the last hidden state to the GRU and
# obtain a current GRU output and hidden state. We then use our ``Attn``
# module as a layer to obtain the attention weights, which we multiply by
# the encoder’s output to obtain our attended encoder output. We use this
# attended encoder output as our ``context`` tensor, which represents a
# weighted sum indicating what parts of the encoder’s output to pay
# attention to. From here, we use a linear layer and softmax normalization
# to select the next word in the output sequence.
#
# Hybrid Frontend Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Similarly to the ``EncoderRNN``, this module does not contain any
# data-dependent control flow. Therefore, we can once again use
# **tracing** to convert this model to Torch Script after it is
# initialized and its parameters are loaded.
#

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


