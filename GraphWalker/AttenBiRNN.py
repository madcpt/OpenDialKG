import json
import csv
import configparser
import sys
import os
import pickle
from tqdm import tqdm
import numpy as np

from nltk.tokenize import word_tokenize

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")

from preprocess.data_reader import load_kg, parse_path_cfg, get_dial_vocab
from preprocess.kg_dataloader import get_kg_DataLoader, get_kg_connection_map, get_two_hops_map
from preprocess.dial_dataloader import get_dial_DataLoader
from KGE.TransE import TransE

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

temp_test_file = './opendialkg/data/tmp.pl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size, rnn_hidden):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.rnn_hidden = rnn_hidden
        self.encoder = nn.GRU(input_size=self.dim, hidden_size=self.rnn_hidden, num_layers=1, batch_first=True,
                              bidirectional=True)

    def forward(self, input_embedding, input_lens, hidden=None):
        input_lens = np.array(input_lens)
        sort_idx = np.argsort(-input_lens)
        input_lengths = input_lens[sort_idx]
        sort_input_seq = input_embedding[sort_idx]
        embedded = sort_input_seq
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, en_hidden = self.encoder(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        invert_sort_idx = np.argsort(sort_idx)
        en_hidden = en_hidden.transpose(0, 1)
        outputs = outputs[invert_sort_idx]
        en_hidden = en_hidden[invert_sort_idx].transpose(0, 1)  # [n*bi , bsz, 400*bi]
        return outputs, en_hidden


if __name__ == '__main__':
    input = torch.randn(16, 20, 300)
