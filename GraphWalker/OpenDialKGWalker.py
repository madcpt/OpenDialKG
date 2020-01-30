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
    def __init__(self, vocab_size, dim, rnn_hidden):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
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


class OpenDialKGWalker(nn.Module):
    def __init__(self, init_kg):
        super().__init__()
        self.embed_dim = 300
        self.rnn_hidden_size = 128
        self.path = parse_path_cfg()
        self.KGEmbeddingModel = TransE(len(entity_map), len(relation_map), 128, 2)
        if init_kg:
            self.KGEmbeddingModel.load_model()
        self.utter_encoder = Encoder(len(word2index), dim=300, rnn_hidden=128)  # TODO attention-bi-gru

        # load word embedding from pre-trained dump
        self.word_embedding = nn.Embedding(len(word2index), 300, padding_idx=1)

        # self.word_embedding.weight.data.normal_(0, 0.1)
        # with open(self.path['DIAL_EMBEDDING'].replace('NUM', str(len(word2index)))) as f:
        #     E = json.load(f)
        # new = self.word_embedding.weight.data.new
        # self.word_embedding.weight.data.copy_(new(E))
        # self.word_embedding.weight.requires_grad = True
        # TODO commented

        self.input_modality_attend = nn.Linear(2 * self.rnn_hidden_size, 2)

    def forward(self, batch_data):
        batch_size = len(batch_data)
        # Input Encoder
        # TODO 3-way modality
        batch_utter = []
        batch_starting_entities_embeddings = []
        for si, sample in enumerate(batch_data):
            # ['dial-id', 'sample-id', 'starting-entities', 'previous-sentence-utter', 'previous-sentence', 'dialogue-history', 'kg-path-id', 'kg-path-search-space']
            batch_utter.append(sample['previous-sentence'])
            starting_entities = torch.tensor(sample['starting-entities']).to(device)
            entities_embeddings = self.KGEmbeddingModel.get_embedding(starting_entities, 'entity')
            entities_embeddings_aggregate = torch.sum(entities_embeddings, dim=0)
            batch_starting_entities_embeddings.append(entities_embeddings_aggregate.unsqueeze(dim=0))
        batch_starting_entities_embeddings = torch.cat(batch_starting_entities_embeddings, dim=0)  # [b, 128]
        input_lens = [len(x) for x in batch_utter]
        max_len = max(input_lens)
        batch_utter = [x + [1] * (max_len - len(x)) for x in batch_utter]
        batch_utter_tensor = torch.tensor(batch_utter).to(device)
        batch_utter_embedding = self.word_embedding(batch_utter_tensor)
        out, hid = self.utter_encoder.forward(batch_utter_embedding, input_lens)  # out: [b, len, 256], hid: [2, b, 128]
        input_aggregate = torch.cat([batch_starting_entities_embeddings.unsqueeze(dim=1), hid[-1].unsqueeze(dim=1)], dim=1)  # [b, 2, 128]
        input_flatten = input_aggregate.reshape(batch_size, -1)
        input_modality_attention_logic = torch.sigmoid(self.input_modality_attend(input_flatten))  # [b, 2]
        input_modality_attention_score = input_modality_attention_logic.softmax(dim=1)
        context_vector = (input_modality_attention_score.unsqueeze(dim=-1) * input_aggregate).sum(dim=1)  # [b, 128]
        return


if __name__ == '__main__':
    entity_map, relation_map, triple_list = load_kg()
    word2index = get_dial_vocab()
    # WholeDataLoader = get_kg_DataLoader(entity_map, relation_map, triple_list)
    connection_map = get_kg_connection_map(entity_map, relation_map, triple_list)
    two_hops_map = get_two_hops_map(connection_map)

    model = OpenDialKGWalker(False)
    with open(temp_test_file, 'rb') as f:
        data = pickle.load(f)
    model(data)
    # print(data)

    # train, dev, test = get_dial_DataLoader(entity_map, relation_map, triple_list, word2index, two_hops_map, 16)
    # for di, data in enumerate(train):
    #     # print(len(data))
    #     with open(temp_test_file, 'wb') as f:
    #         pickle.dump(data, f)
    #     break
