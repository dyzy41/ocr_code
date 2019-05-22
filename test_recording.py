from __future__ import print_function
import argparse
import random
import torch
import re
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import dataset
from collections import OrderedDict
from model_resnet import CNN, EncoderRNN, LuongAttnDecoderRNN, model
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--valroot', default='/home/wangxiang/Documents/ocr_data/data_cvt_gray/val', help='path to dataset')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpu_list = [0]

MAX_LENGTH = 40

PAD_token = 0
SOS_token = 1
EOS_token = 2

imgW = 512
imgH = 64
batchSize = 40
workers = 2

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


def evaluate(model, image, max_length=MAX_LENGTH):
    with torch.no_grad():
        outputs = model(image, max_length)
        outputs = outputs.permute(1, 0, 2)

        decoded_words = []
        for cur_output in outputs:
            _, cur_tokens = torch.max(cur_output, dim=1)
            decoded_words.append([index2word[cur_tensor.item()] for cur_tensor in cur_tokens])

        return np.array(decoded_words).T


def equal(input_sentence, output_sentence):
    if input_sentence == output_sentence.replace('EOS', '').replace('PAD', '').replace(' ', ''):
        return True


def test(model):
    val_iter = iter(test_loader)
    right = 0
    all = 0
    for _ in tqdm(range(len(test_loader))):
        data = val_iter.next()
        cpu_image, cpu_text = data
        all = all + len(cpu_image)
        image = cpu_image.cuda()

        output_words = evaluate(model, image)
        output_sentences = [' '.join(cur_words) for cur_words in output_words]

        for input_sentence, output_sentence in zip(cpu_text, output_sentences):
            if equal(input_sentence.decode('utf-8'), output_sentence):
                right += 1

    return float(right)/all


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(os.path.join(dir, fn))
    if not os.path.isdir(os.path.join(dir, fn)) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


def load_modelpth(model, model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    print('load the model %s' % model_path)
    model_id = re.findall(r'\d+', model_path)
    model_id = int(model_id[0])

    return model, model_id


if __name__ == '__main__':
    model_dir = './expr/model'

    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, transform=dataset.resizeNormalize((imgW, imgH)))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, sampler=None, batch_size=batchSize, num_workers=int(workers))

    word2index, index2word, n_words = lang()

    nc = 1
    input_size = 1024
    hidden_size = 1024
    encoder_layers = 2
    decoder_layers = 1
    attn_model = 'general'
    embedding = nn.Embedding(n_words, hidden_size)

    cnn = CNN(imgH, nc)
    encoder = EncoderRNN(input_size, hidden_size, n_layers=encoder_layers, dropout=0)
    attn_decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, n_words, n_layers=decoder_layers, dropout=0)
    model = model(cnn, encoder, attn_decoder).cuda()
    ff = open('recording.txt', 'r', encoding='utf-8')
    lines = ff.readlines()
    x = lines[-1]
    last_id = x.split(':')[0]
    last_model = 'model_Rec_done_{}.pth'.format(last_id)
    ff.close()

    with open('recording.txt', 'a', encoding='utf-8') as f:
        if os.path.exists(model_dir):
            path = os.listdir(model_dir)
            path.sort(key=lambda i: int(re.findall(r'\d+', i)[0]))
            p = path.index(last_model)
            path_ = path[p+1:]

            for cur_path in tqdm(path_):
                cur_model_path = os.path.join(model_dir, cur_path)
                model, model_id = load_modelpth(model, cur_model_path)
                acc = test(model)
                f.writelines((str(model_id) + ':').ljust(6) + str(acc) + '\n')

