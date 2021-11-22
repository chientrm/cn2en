from __future__ import unicode_literals, print_function, division
from pathlib import Path
import string
import random
import math
from functools import reduce

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

        
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Lang':
            return Lang
        return super().find_class(module, name)


def normalize(s):
    return ''.join([c if c in 'abcdefghijklmnopqrstuvwxyz\'" '
        else ' ' + c + ' ' for c in s.lower().strip()]).strip()


def tensorsFromSentence(lang, sentence):
    indices = [lang.word2index[w] 
               for w in sentence.split(' ') if w and w in lang.word2index
    ] + [EOS_token]
    return list(map(lambda index: torch.tensor([index], dtype=torch.long, device=device), indices))


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


def translate(input, encoder, decoder):
    with torch.no_grad():
        h = reduce(lambda h, w: encoder(w, h)[1], [encoder.initHidden()] + input)

        decoder_input = torch.tensor([[SOS_token]], device=device)

        outputs = []
        for i in range(1000):
            output, h = decoder(decoder_input, h)
            topv, topi = output.data.topk(1)
            if topi.item() == EOS_token:
                break
            outputs.append(topi.item())
            decoder_input = topi.squeeze().detach()
        return outputs


class Model():
    def __init__(self):
        self.hidden_size = 256
        with Path(__file__).with_name('input_lang.pkl').open('rb') as f:
            self.input_lang = CustomUnpickler(f).load()
        with Path(__file__).with_name('output_lang.pkl').open('rb') as f:
            self.output_lang = CustomUnpickler(f).load()
        self.encoder = EncoderRNN(self.input_lang.n_words, self.hidden_size).to(device)
        self.decoder = DecoderRNN(self.hidden_size, self.output_lang.n_words).to(device)
        checkpoint = torch.load(Path(__file__).with_name('model.pth'),
                                map_location=torch.device('cpu'))
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])


    def translate(self, sentence):
        outs = translate(tensorsFromSentence(self.input_lang, normalize(sentence)),
                         self.encoder, self.decoder)
        return ' '.join(list(map(lambda i: self.output_lang.index2word[i], outs))).strip()