import json
import csv
import configparser
import sys
import os
from tqdm import tqdm

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")

from preprocess.data_reader import load_kg

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
        self.triple_ind_list = [
            (entity_map[t.split('\t')[0]], relation_map[t.split('\t')[1]], entity_map[t.split('\t')[2]]) for t in
            triple_list]

    def __len__(self):
        return len(self.triple_list)

    def __getitem__(self, item):
        # return {'head': self.triple_ind_list[item][0], 'relation': self.triple_ind_list[item][1], 'tail': self.triple_ind_list[item][2]}
        return {'triple': self.triple_ind_list[item]}


def get_kg_connection_map(entity_map, relation_map, triple_list):
    connection_map = {entity_id: {} for entity_id, _ in enumerate(entity_map)}
    for triple in triple_list:
        s_r_t = triple.split('\t')
        si = entity_map[s_r_t[0]]
        ri = relation_map[s_r_t[1]]
        ti = entity_map[s_r_t[2]]
        if ri not in connection_map[si]:
            connection_map[si][ri] = [ti]
        else:
            connection_map[si][ri].append(ti)
    return connection_map


def get_two_hops_map(connection_map):
    raise DeprecationWarning('Please use get-two-hops_connection function!')
    two_hops_map = {entity_id: [] for entity_id in connection_map}
    for start in connection_map:
        for r1, t1s in connection_map[start].items():
            for t1 in t1s:
                two_hops_map[start].append((r1, t1))
                for r2, t2s in connection_map[t1].items():
                    for t2 in t2s:
                        two_hops_map[start].append((r1, t1, r2, t2))
    return two_hops_map


def get_two_hop_paths(start, connection_map):
    two_hop_paths = []
    for r1, t1s in connection_map[start].items():
        for t1 in t1s:
            two_hop_paths.append((start, r1, t1))
            for r2, t2s in connection_map[t1].items():
                for t2 in t2s:
                    two_hop_paths.append((start, r1, t1, r2, t2))
    return two_hop_paths


if __name__ == '__main__':
    entity_map, relation_map, triple_list = load_kg()
    WholeDataLoader = get_kg_DataLoader(entity_map, relation_map, triple_list)
    connection_map = get_kg_connection_map(entity_map, relation_map, triple_list)
    # two_hops_map = get_two_hops_map(connection_map)
    two_hop_paths_demo = get_two_hop_paths(0, connection_map)

    # for k, v in two_hops_map.items():
    #     print(k, v)
    #     break

    # for k in connection_map:
    #     print(k, connection_map[k])
    #     break
    #
    # for data in WholeDataLoader:
    #     print(data)
    #     break
