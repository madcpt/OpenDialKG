import json
import csv
import configparser
import sys
import os
from tqdm import tqdm

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")

from preprocess.create_data import load_kg

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


def parse_load_cfg():
    cfg = configparser.ConfigParser()
    cfg.read("./preprocess/dataset.cfg")
    return cfg['LOAD']


def get_kg_DataLoader(entity_map, relation_map, triple_list, batch_size=None):
    if batch_size is None:
        load_cfg = parse_load_cfg()
        batch_size = int(load_cfg['BATCH'])
    WholeDataSet = KGData(triple_list, entity_map, relation_map)
    WholeDataLoader = DataLoader(WholeDataSet, batch_size=batch_size, shuffle=True)
    return WholeDataLoader


class KGData(Dataset):
    # noinspection PyShadowingNames
    def __init__(self, triple_list, entity_map, relation_map):
        self.triple_list = triple_list
        self.triple_ind_list = [(entity_map[t.split('\t')[0]], relation_map[t.split('\t')[1]], entity_map[t.split('\t')[2]]) for t in triple_list]

    def __len__(self):
        return len(self.triple_list)

    def __getitem__(self, item):
        # return {'head': self.triple_ind_list[item][0], 'relation': self.triple_ind_list[item][1], 'tail': self.triple_ind_list[item][2]}
        return {'triple': self.triple_ind_list[item]}


if __name__ == '__main__':
    entity_map, relation_map, triple_list = load_kg()
    WholeDataLoader = get_kg_DataLoader(entity_map, relation_map, triple_list)

    for data in WholeDataLoader:
        print(data)
        exit()
