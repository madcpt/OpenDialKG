import configparser
import sys
import os
import argparse
from tqdm import tqdm

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")

from preprocess.create_data import load_kg
from preprocess.kg_dataloader import get_kg_DataLoader

import torch
from torch import nn

parser = argparse.ArgumentParser(description='Knowledge Graph Embedding')

# Training Setting
parser.add_argument('-b', '--batch', help='batch size', type=int, required=False, default=16)
parser.add_argument('-lr', '--lr', help='learning rate', type=float, required=False, default=1e-3)
parser.add_argument('-e', '--embedding', help='embedding size', type=int, required=False, default=128)
parser.add_argument('-g', '--gamma', help='parameter gamma', type=int, required=False, default=2)
parser.add_argument('-save', '--save_path', help='save path for KGE', type=str, required=False, default='./save/KGE')
parser.add_argument('-cuda', '--use_cuda', help='Use cuda', type=int, required=False, default=1)

args = vars(parser.parse_args())

device = torch.device('cuda' if args['use_cuda'] else 'cpu')


class TransE(nn.Module):
    def __init__(self, entity_size, relation_size, embedding_size, gamma, add_name='transe'):
        super().__init__()
        self.EntityEmbedding = nn.Embedding(entity_size, embedding_size)
        self.RelationEmbedding = nn.Embedding(relation_size, embedding_size)
        self.EntityEmbedding.weight.data.normal_(0.1)
        self.RelationEmbedding.weight.data.normal_(0.1)
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args['lr'])
        self.save_path = "%s/%s" % (args['save_path'], add_name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.entity_path = "%s/%s/entity.th" % (args['save_path'], add_name)
        self.relation_path = "%s/%s/relation.th" % (args['save_path'], add_name)

    def negative_sampling(self, shape):
        negative_head = torch.randint(high=self.entity_size, size=shape, device=device)
        negative_tail = torch.randint(high=self.entity_size, size=shape, device=device)
        return negative_head, negative_tail

    def get_embedding(self, input_tensor, type='entity'):
        assert type in ['entity', 'relation']
        if type == 'entity':
            input_embedding = self.EntityEmbedding(input_tensor)
        else:
            input_embedding = self.RelationEmbedding(input_tensor)
        return nn.functional.normalize(input_embedding, 2, 1)

    @staticmethod
    def distance(head_embedding, rel_embedding, tail_embedding):
        return nn.functional.pairwise_distance(head_embedding + rel_embedding, tail_embedding, p=2)

    def calc_loss(self, head_embedding, rel_embedding, tail_embedding, n_head_embedding, n_tail_embedding, gamma=0.1):
        d1 = self.distance(head_embedding, rel_embedding, tail_embedding)
        d2 = self.distance(n_head_embedding, rel_embedding, tail_embedding)
        return nn.functional.relu(self.gamma + d1 - d2).sum()

    def forward(self, data):
        triple_tensor = data['triple']
        head, rel, tail = triple_tensor
        head = head.to(device)
        rel = rel.to(device)
        tail = tail.to(device)
        negative_head, negative_tail = self.negative_sampling(head.shape)
        head_embedding = self.get_embedding(head, type='entity')
        rel_embedding = self.get_embedding(rel, type='relation')
        tail_embedding = self.get_embedding(tail, type='entity')
        n_head_embedding = self.get_embedding(negative_head, type='entity')
        n_tail_embedding = self.get_embedding(negative_tail, type='entity')
        loss = self.calc_loss(head_embedding, rel_embedding, tail_embedding, n_head_embedding, n_tail_embedding)
        return loss

    def run_batch(self, data, optimize):
        self.train()
        loss = self.forward(data)
        if optimize:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def run_epoch(self, data_loader, optimize):
        if optimize:
            self.train()
        else:
            self.eval()
        pbar = tqdm(data_loader, total=len(data_loader))
        loss, cnt = 0, 0
        for di, data in enumerate(pbar):
            if optimize:
                batch_loss = model.run_batch(data, optimize)
            else:
                with torch.no_grad():
                    batch_loss = model.run_batch(data, optimize)
            cnt += data['triple'][0].size(0)
            loss += batch_loss
            pbar.set_description('l: %.4f' % (loss / cnt))
        return loss/cnt

    def save_model(self):
        print("Saving to %s" % self.save_path)
        torch.save(self.EntityEmbedding.state_dict(), self.entity_path)
        torch.save(self.RelationEmbedding.state_dict(), self.relation_path)

    def load_model(self):
        print("Loading from %s" % self.save_path)
        self.EntityEmbedding.load_state_dict(torch.load(self.entity_path))
        self.RelationEmbedding.load_state_dict(torch.load(self.relation_path))


if __name__ == '__main__':
    entity_map, relation_map, triple_list = load_kg()
    WholeDataLoader = get_kg_DataLoader(entity_map, relation_map, triple_list, int(args['batch']))

    model = TransE(entity_size=len(entity_map), relation_size=len(relation_map), embedding_size=args['embedding'],
                   gamma=args['gamma']).to(device)

    # model.save_model()
    model.load_model()

    best_loss = 10000
    patient = 5
    for epoch in range(20):
        if patient <= 0:
            print('ran out of patient')
            break
        train_loss = model.run_epoch(WholeDataLoader, optimize=True)
        # print('TRAIN - epoch %d: %.4f' % (epoch, train_loss))
        test_loss = model.run_epoch(WholeDataLoader, optimize=False)
        print('TEST - epoch %d: %.4f' % (epoch, test_loss))
        if test_loss < best_loss:
            best_loss = test_loss
            model.save_model()
            patient = 5
        else:
            patient -= 1
