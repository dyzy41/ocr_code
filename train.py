from __future__ import print_function
import argparse
import random
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# from torch.autograd import Variable
import numpy as np
import os
import time
from collections import OrderedDict
import math
# import utils
import dataset
# import params as paramsaehs-
from tensorboardX import SummaryWriter
# from model_multi_gpu import CNN, CNN_res, EncoderRNN, LuongAttnDecoderRNN, model
# from new_model import CNN, EncoderRNN, LuongAttnDecoderRNN, model
from model_resnet import CNN, EncoderRNN, LuongAttnDecoderRNN, model

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='../ocr_data/data/train', help='path to dataset')
parser.add_argument('--valroot', default='../ocr_data/data/val', help='path to dataset')
# parser.add_argument('--trainroot', default='/home/weikai/Documents/ocr_data/data_small/train', help='path to dataset')
# parser.add_argument('--valroot', default='/home/weikai/Documents/ocr_data/data_small/val', help='path to dataset')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_list = [0]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 40

PAD_token = 0
SOS_token = 1
EOS_token = 2

base_lr = 0.001
lr = base_lr
weight_decay = 2e-5
momentum = 0.9
power = 3
min_lr = 0.0001

random_sample = True
keep_ratio = False
adam = False
adadelta = False
saveInterval = 2
valInterval = 2

n_test_disp = 10
num_save_model = 80
displayInterval = 5
experiment = './expr'
crnn = ''
beta1 = 0.5
niter = 10000
nh = 256
imgW = 512
imgH = 64
batchSize = 5

workers = 2

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def lang():
    word2index = {}
    index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
    n_words = 3
    with open("alphabet.txt", 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.strip("\n"):
                word2index[word] = n_words
                index2word[n_words] = word
                n_words += 1
    return word2index, index2word, n_words

def indexesFromSentence(word2index, sentence):
    indexes = [word2index[word] for word in sentence]
    indexes.append(EOS_token)
    return indexes

def tensorFromSentence(word2index, sentence):
    indexes = indexesFromSentence(word2index, sentence)
    # indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).cuda().view(-1, 1)

def train(image, text, model, model_optimizer, criterion):
    loss = 0
    model.zero_grad()

    target_index = [indexesFromSentence(word2index, cur_text.decode('utf-8')) for cur_text in text]
    target_maxlength = len(max(target_index, key=lambda tar: len(tar)))

    for i in range(len(target_index)):
        m = target_maxlength - len(target_index[i])
        if m != 0:
            target_index[i].extend(np.zeros(m, dtype=np.int))

    target_tensor = torch.tensor(target_index, dtype=torch.long).cuda().permute(1, 0)

    output = model(image, target_maxlength)
    decoder_outputs = output.permute(1, 0, 2)

    for decoder_output, cur_target_tensor in zip(decoder_outputs, target_tensor):
        loss += criterion(decoder_output, cur_target_tensor)

    loss.backward()
    model_optimizer.step()

    return loss.item() / target_maxlength

def evaluate(model, image, max_length=MAX_LENGTH):
    with torch.no_grad():
        outputs = model(image, max_length)
        outputs = outputs.permute(1, 0, 2)

        decoded_words = []
        for cur_output in outputs:
            _, cur_tokens = torch.max(cur_output, dim=1)
            decoded_words.append([index2word[cur_tensor.item()] for cur_tensor in cur_tokens])

        return np.array(decoded_words).T

def adjust_learning_rate(base_lr, model_optimizer, epoch, num_epochs, power):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * ((1-float(epoch)/num_epochs)**power) + min_lr
    for param_group in model_optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def evaluateRandomly(model, max_iter=1):

    val_iter = iter(test_loader)

    max_iter = min(max_iter, len(test_loader))
    for i in range(max_iter):
        data = val_iter.next()
        cpu_image, cpu_text = data
        image = cpu_image.cuda()

        output_words = evaluate(model, image)
        output_sentences = [' '.join(cur_words) for cur_words in output_words]
        # output_sentence = output_words

        for input_sentence, output_sentence in zip(cpu_text, output_sentences):
            print('>', input_sentence.decode('utf-8'))
            print('<', output_sentence.replace('EOS', '').replace('PAD', ''))
            print('')

def trainIters(model, criterion, plot_every=100):

    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0

    model_optimizer = optim.SGD(model.parameters(),lr=base_lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(1, niter + 1):
        lr = adjust_learning_rate(base_lr, model_optimizer, epoch+model_id, niter, power)
        train_iter = iter(train_loader)
        print(len(train_loader))
        all_loss = 0
        for i in range(1, len(train_loader) + 1):
            data = train_iter.next()
            cpu_images, cpu_texts = data
            text = cpu_texts
            image = cpu_images.cuda()
            loss = train(image, text, model, model_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
            all_loss += loss

            if i % displayInterval == 0:
                print_loss_avg = print_loss_total / displayInterval
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, i / niter),
                                             epoch+model_id, (epoch+model_id) / niter * 100, print_loss_avg))

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                writer.add_scalar('lost', plot_loss_avg, (epoch+model_id - 1) * len(train_loader) + i - 1)
                plot_loss_total = 0

        writer.add_scalar('all_loss', all_loss / len(train_loader), (epoch+model_id - 1))
        writer.add_scalar('lr', lr, (epoch+model_id - 1))
        if (epoch+model_id) % valInterval == 0:
            evaluateRandomly(model)

        if (epoch+model_id) % saveInterval == 0:
            torch.save(model.state_dict(), '{0}/model_Rec_done_{1}.pth'.format(experiment + '/model', epoch+model_id))
            print('save ' + '{0}/model_Rec_done_{1}.pth'.format(experiment + '/model', epoch+model_id))

        if (epoch+model_id) % (saveInterval*(num_save_model+2)) == 0:
            path = './expr/model/'
            delete_models([path])

def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
                    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None

def delete_models(path):
    for i in range(len(path)):
        model_name = os.listdir(path[i])
        model_name.sort(key=lambda x: os.path.getctime(os.path.join(path[i], x)))
        del_name = model_name[0:-num_save_model]
        for f in del_name:
            os.remove(path[i] + f)

def load_modelpth(model):
    model_dir = './expr/model/'
    if find_new_file(model_dir) is not None:
        state_dict = torch.load(find_new_file(model_dir))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        print('load the model %s' % find_new_file(model_dir))
        model_id = re.findall(r'\d+', find_new_file(model_dir))
        model_id = int(model_id[0])
    return model, model_id

if __name__ == '__main__':
    writer = SummaryWriter()
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    cudnn.benchmark = True

    if not os.path.exists('./expr/model'):
        os.makedirs('./expr/model')

    train_dataset = dataset.lmdbDataset(root=opt.trainroot)
    assert train_dataset
    if not random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, batchSize)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchSize,
        shuffle=True, sampler=sampler,
        num_workers=int(workers),
        collate_fn=dataset.alignCollate(imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, transform=dataset.resizeNormalize((imgW, imgH)))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, sampler=None, batch_size=50, num_workers=int(workers))

    word2index, index2word, n_words = lang()

    nc = 1
    input_size = 1024
    hidden_size = 1024
    encoder_layers = 2
    decoder_layers = 1
    attn_model = 'general'
    embedding = nn.Embedding(n_words, hidden_size)
    criterion = nn.NLLLoss()

    model_id = 0
    cnn = CNN(imgH, nc)
    # cnn = CNN_res(imgH)
    encoder = EncoderRNN(input_size, hidden_size, n_layers=encoder_layers)
    attn_decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, n_words, n_layers=decoder_layers)
    model = model(cnn, encoder, attn_decoder)
    # if os.listdir('./expr/cnn') != []:
    #     cnn, model_id = load_cnnpth(cnn)
    #     encoder, _ = load_encoderpth(encoder)
    #     attn_decoder, _ = load_decoderpth(attn_decoder)
    if os.listdir('./expr/model') != []:
        model, model_id = load_modelpth(model)

    model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()
    trainIters(model, criterion)
    writer.close()


