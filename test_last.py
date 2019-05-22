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

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='/home/kawhi/Documents/ocr_data/data_cvt_gray/train', help='path to dataset')
parser.add_argument('--valroot', default='/home/wangxiang/Documents/ocr_data/testimg_cvt_gray/test', help='path to dataset')
#parser.add_argument('--valroot', default='/home/wangxiang/Documents/ocr_data/testimg_hsv/test', help='path to dataset')
# parser.add_argument('--valroot', default='/home/weikai/Documents/ocr_data/data_cvt_gray/train', help='path to dataset')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


def evaluateRandomly(model):
    while True:
        val_iter = iter(test_loader)
        for i in range(len(test_loader)):
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

            answer = input("\nDo you want continue(Y/N)")
            if answer.upper() == 'N':
                exit(0)


def equal(input_sentence, output_sentence):
    if input_sentence == output_sentence.replace('EOS', '').replace('PAD', '').replace(' ', ''):
        return True


def test(model):
    val_iter = iter(test_loader)
    print(len(test_loader))
    with open('right.txt', 'w', encoding='utf-8') as f:
        right = 0
        all = 0
        pred = open('pred.txt', 'w', encoding='utf-8')
        for i in range(len(test_loader)):

            data = val_iter.next()
            cpu_image, cpu_text = data
            all = all + len(cpu_image)
            image = cpu_image.cuda()

            output_words = evaluate(model, image)
            output_sentences = [' '.join(cur_words) for cur_words in output_words]
            # output_sentence = output_words

            for input_sentence, output_sentence in zip(cpu_text, output_sentences):
                print(output_sentence.replace('EOS', '').replace('PAD', '').replace(' ', ''))
                pred.writelines(output_sentence.replace('EOS', '').replace('PAD', '').replace(' ', '') + '\n')
                if equal(input_sentence.decode('utf-8'), output_sentence):
                    right += 1
                    print(right)
                # if not equal(input_sentence.decode('utf-8'), output_sentence):
                #     print(output_sentence.replace('EOS', '').replace('PAD', '').replace(' ', ''))
                    f.writelines(input_sentence.decode('utf-8') + "\n")
        print('the right is %d' % right)
        print('the all is %d' % all)
            # print('%d of %d' % (i, len(test_loader)))
    return float(right)/all


def out(model, model_id):
    val_iter = iter(test_loader)
    print(len(test_loader))
    file_name = 'recognition_result_' + str(model_id) + '.txt' 
    with open(file_name, 'w', encoding='utf-8') as f:
        result_list = []
        for i in range(len(test_loader)):
            data = val_iter.next()
            cpu_image, cpu_text = data

            image = cpu_image.cuda()

            output_words = evaluate(model, image)
            output_sentences = [' '.join(cur_words) for cur_words in output_words]

            for input_sentence, output_sentence in zip(cpu_text, output_sentences):
                print(output_sentence.replace('EOS', '').replace('PAD', '').replace(' ', ''))
                result_list.append(input_sentence.decode('utf-8').ljust(10) + output_sentence.replace('EOS', '').replace('PAD', '').replace(' ', '') + "\n")
        result_list = sorted(result_list, key=lambda i: int(i.split(' ')[0].split('_')[1]))
        result_list = sorted(result_list, key=lambda i: int(i.split(' ')[0].split('_')[0]))
        f.writelines(result_list)


def find_new_file(dir):
    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(os.path.join(dir, fn))
    if not os.path.isdir(os.path.join(dir, fn)) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None


def load_modelpth(model):
    model_dir = './expr/model/'
    #path = find_new_file(model_dir)
    path = '/home/wangxiang/Documents/full_multi_gpu/expr/model/model_Rec_done_1078.pth'
    if find_new_file(model_dir) is not None:
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        print('load the model %s' % path)
        model_id = re.findall(r'\d+', path)
        model_id = int(model_id[0])
    return model, model_id

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
encoder = EncoderRNN(input_size, hidden_size, n_layers=encoder_layers,dropout=0)
attn_decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, n_words, n_layers=decoder_layers)
model = model(cnn, encoder, attn_decoder)
if os.path.exists('./expr/model'):
    model, model_id = load_modelpth(model)
model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()
# def eval_test():
#     acc = test(model)
#     return acc
#     # evaluateRandomly(model)
# accuracy = []
# for i in range(2):
#     x = eval_test()
#     accuracy.append(x)
#     print(x)
# print(accuracy)
out(model, model_id)
