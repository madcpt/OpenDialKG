import csv
import configparser
import sys
import random
import os
import json

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")


def split_opendialkg():
    """
    Read the dialogue part of the raw dataset, then randomly split the dialogues into [0.7 : 0.15 : 0.15]
    """
    cfg = configparser.ConfigParser()
    cfg.read("./preprocess/dataset.cfg")
    path = cfg['PATH']
    dials = []
    with open(path['DIALOGUE']) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            content = json.loads(row[0])
            dials.append({'dial_id': i - 1, 'dialogue': content})
    random.shuffle(dials)
    train_num = int(len(dials) * 0.7)
    dev_num = int(len(dials) * 0.15)
    test_num = len(dials) - train_num - dev_num
    print('Train: ', train_num)
    print('Dev: ', dev_num)
    print('Test: ', test_num)
    train_dials = dials[:train_num]
    dev_dials = dials[train_num: train_num + dev_num]
    test_dials = dials[train_num + dev_num:]
    if not os.path.exists(path['TRAIN_FILE']) or not os.path.exists(path['DEV_FILE']) or not os.path.exists(
            path['TEST_FILE']):
        with open(path['TRAIN_FILE'], 'w') as f:
            json.dump(train_dials, f)
        with open(path['DEV_FILE'], 'w') as f:
            json.dump(dev_dials, f)
        with open(path['TEST_FILE'], 'w') as f:
            json.dump(test_dials, f)
    else:
        raise FileExistsError('Dataset split file already existed.')


if __name__ == '__main__':
    split_opendialkg()
