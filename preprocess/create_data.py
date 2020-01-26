import json
import csv
import configparser
import sys
import os

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")


def read_raw():
    from preprocess.fix_dataset_error import (entity_list, relation_list, triple_list)

    cfg = configparser.ConfigParser()
    cfg.read("./preprocess/path.cfg")
    path = cfg['PATH']

    with open(path['ENTITY'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            entity_list.append(line.strip('\n'))
    entity_map = {v: i for i, v in enumerate(entity_list)}
    print('entities: ', len(entity_map))

    with open(path['RELATION'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            relation_list.append(line.strip('\n'))
    relation_map = {v: i for i, v in enumerate(relation_list)}
    print('relations: ', len(relation_map))

    triple_error = 0
    with open(path['TRIPLE'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            s, r, t = line.strip('\n').split('\t')
            try:
                # if r[0] == '~':
                #     triple_list.append((entity_map[t], relation_map[r[1:]], entity_map[s]))
                # else:
                #     triple_list.append((entity_map[s], relation_map[r], entity_map[t]))
                assert s in entity_map
                assert r in relation_map
                assert t in entity_map
                triple_list.append(line.strip('\n'))
            except KeyError:
                triple_error += 1
    print('triples: ', len(triple_list))
    # print('triple errors: ', triple_error)
    return entity_map, relation_map, triple_list


def read_dial(dial_path, entity_map, relation_map, triple_list):
    bad_entity = 0
    bad_rel = 0
    bad_path = 0
    with open(dial_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            content = json.loads(row[0])
            for turn in content:
                # print(action_ids)
                if 'metadata' in turn and 'path' in turn['metadata']:
                    score, path, utterance = turn['metadata']['path']
                    # if len(path) > 1:
                    #     print(path)
                    #     exit()
                    for triple in path:
                        if '\t'.join(triple) not in triple_ori:
                            # and '\t'.join(triple).replace('~', '') not in triple_ori:
                            bad_path += 1
                            print(triple)
                            # print(p in triple_ori)
                        if triple[0] not in entity_map:
                            print(triple[0])
                            bad_entity += 1
                        if triple[1] not in relation_map:
                            print(triple[1])
                            bad_rel += 1
                        if triple[2] not in entity_map:
                            print(triple[2])
                            bad_entity += 1
            # user_rating = json.loads(row[1])
            # assistant_rating = json.loads(row[2])
            # print(user_rating)
            # print(assistant_rating)

    # print('bad entity: ', bad_entity)
    # print('bad relation: ', bad_rel)
    # print('bad path: ', bad_path)
    print('finish')
    return
