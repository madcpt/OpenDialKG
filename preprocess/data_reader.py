import json
import csv
import configparser
import sys
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from embeddings import GloveEmbedding

module_path = os.path.abspath('.')
sys.path.insert(-1, module_path)
sys.path.append("../../")


def dump_pretrained_emb(word2index, dump_path):
    print("Dumping pretrained embeddings into ", dump_path)
    index2word = {v: k for k, v in word2index.items()}
    embeddings = [GloveEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def parse_path_cfg():
    cfg = configparser.ConfigParser()
    cfg.read("./preprocess/dataset.cfg")
    path = cfg['PATH']
    return path


def load_kg():
    from preprocess.fix_dataset_error import (entity_list, relation_list, triple_list)
    path = parse_path_cfg()

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

    with open(path['TRIPLE'], 'r', encoding='utf-8') as f:
        for line in f.readlines():
            s, r, t = line.strip('\n').split('\t')
            try:
                assert s in entity_map
                assert r in relation_map
                assert t in entity_map
                triple_list.append(line.strip('\n'))
            except KeyError:
                raise KeyError('Error encountered at ', line.strip('\n'))
    print('triples: ', len(triple_list))
    return entity_map, relation_map, triple_list


def load_dials(dial_type, entity_map, relation_map, triple_list):
    assert dial_type in ['train', 'dev', 'test']
    path = parse_path_cfg()
    triple_set = set(triple_list)
    dial_file_path = path['%s_FILE' % dial_type.upper()]
    print('Loading from ', dial_file_path, end=' ... ')
    with open(dial_file_path, 'r') as f:
        dataset = json.load(f)
        print('Size: ', len(dataset), end=' ... ')
        for dial in tqdm(dataset, total=len(dataset), disable=True):
            content = dial['dialogue']
            for turn in content:
                # print(action_ids)
                if 'metadata' in turn and 'path' in turn['metadata']:
                    score, path, utterance = turn['metadata']['path']
                    # if len(path) > 1:
                    #     print(path)
                    #     exit()
                    for triple in path:
                        if '\t'.join(triple) not in triple_set:
                            raise Exception('Unexpected triple: ', triple)
                        if triple[0] not in entity_map:
                            raise Exception('Unexpected entity: ', triple[0])
                        if triple[1] not in relation_map:
                            raise Exception('Unexpected relation: ', triple[1])
                        if triple[2] not in entity_map:
                            raise Exception('Unexpected entity: ', triple[2])
            # user_rating = json.loads(row[1])
            # assistant_rating = json.loads(row[2])
            # print(user_rating)
            # print(assistant_rating)
    print('%s: finish' % dial_type.upper())
    return


def get_dial_vocab():
    word2index = {'UNK': 0, 'PAD': 1, 'EOS': 2}
    path = parse_path_cfg()
    if os.path.exists(path['DIAL_VOCAB']):
        print('Loading from ', path['DIAL_VOCAB'])
        with open(path['DIAL_VOCAB'], 'r', encoding='utf-8') as f:
            word2index = json.load(f)
    else:
        for dial_type in ['train', 'dev', 'test']:
            dial_file_path = path['%s_FILE' % dial_type.upper()]
            with open(dial_file_path, 'r') as f:
                dataset = json.load(f)
                for sample in tqdm(dataset, total=len(dataset), disable=True):
                    dial = sample['dialogue']
                    for turn in dial:
                        if 'action_id' in turn and turn['action_id'] == 'kgwalk/choose_path':
                            utter = turn['metadata']['path'][2]
                        elif 'action_id' in turn and turn['action_id'] == 'meta_thread/send_meta_message':
                            utter = ''
                        else:
                            utter = turn['message']
                        tokens = word_tokenize(utter)
                        # print(tokens)
                        for word_ in tokens:
                            word = word_.lower()
                            if word not in word2index:
                                word2index[word] = len(word2index)
        with open(path['DIAL_VOCAB'], 'w', encoding='utf-8') as f:
            json.dump(word2index, f)
    print('Vocab Size: ', len(word2index))
    dump_path = path['DIAL_EMBEDDING'].replace('NUM', str(len(word2index)))
    if not os.path.exists(dump_path):
        dump_pretrained_emb(word2index, dump_path)
    return word2index


if __name__ == '__main__':
    entity_map, relation_map, triple_list = load_kg()
    load_dials('train', entity_map, relation_map, triple_list)
    load_dials('dev', entity_map, relation_map, triple_list)
    load_dials('test', entity_map, relation_map, triple_list)

    get_dial_vocab()
