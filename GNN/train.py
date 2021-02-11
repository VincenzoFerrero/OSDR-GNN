import os
import time
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from dataset import DataSet
from itertools import product
from gnn import HierarchicalGNN
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score


class HierarchicalClassifier(object):
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_layers = args.num_layers
        self.num_class_l1 = args.num_class_l1
        self.num_class_l2 = args.num_class_l2
        self.num_class_l3 = args.num_class_l3
        self.patience = args.patience
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.hid_dim = args.hid_dim
        self.lr = args.lr

        self.gnn = HierarchicalGNN(self.node_dim, self.edge_dim, self.hid_dim, self.num_class_l1,
                                   self.num_class_l2, self.num_class_l3, self.num_layers, args.network).to(self.device)

    def load(self):
        if os.path.exists('checkpoint.pkl'):
            self.gnn.load_state_dict(torch.load('checkpoint.pkl'))
        else:
            raise Exception('Checkpoint not found ...')

    def train(self, train_loader, val_loader, weights):
        best_loss, best_state, patience_count = 1e9, None, 0
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        weights_l1 = weights[1].to(self.device)
        weights_l2 = weights[2].to(self.device)
        weights_l3 = weights[3].to(self.device)

        for epoch in range(self.num_epochs):
            self.gnn.train()
            epoch_loss = 0.
            start = time.time()
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                logits_l1, logits_l2, logits_l3 = self.gnn(
                    batch.x.float(), batch.edge_index, batch.e.float(),
                    F.one_hot(batch.y1, self.num_class_l1),
                    F.one_hot(batch.y2, self.num_class_l2))

                is_labeled = batch.y1 > 0
                loss1 = nn.CrossEntropyLoss(weight=weights_l1)(logits_l1[is_labeled], batch.y1[is_labeled])
                is_labeled = batch.y2 > 0
                loss2 = nn.CrossEntropyLoss(weight=weights_l2)(logits_l2[is_labeled], batch.y2[is_labeled])
                is_labeled = batch.y3 > 0
                loss3 = nn.CrossEntropyLoss(weight=weights_l3)(logits_l3[is_labeled], batch.y3[is_labeled])
                loss = loss1 + loss2 + loss3
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            scheduler.step()
            end = time.time()
            val_loss, _, _, _, _, _, _ = self.predict(val_loader)

            # print(f'Epoch: {epoch + 1:03d}/{self.num_epochs}, Time: {end-start:.2f}s, '
            #       f'Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss: .4f}')

            if best_loss > val_loss:
                best_loss = val_loss
                best_state = self.gnn.state_dict()
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == self.patience:
                # print('Early stopping ...')
                break

        self.gnn.load_state_dict(best_state)
        torch.save(best_state, 'checkpoint.pkl')

    @torch.no_grad()
    def predict(self, data_loader):
        self.gnn.eval()

        loss = 0.
        yp_l1, yp_l2, yp_l3 = [], [], []
        yt_l1, yt_l2, yt_l3 = [], [], []

        for batch in data_loader:
            batch = batch.to(self.device)
            logits_l1, logits_l2, logits_l3 = self.gnn.predict(batch.x.float(), batch.edge_index, batch.e.float())
            is_labeled = batch.y1 > 0
            loss1 = nn.CrossEntropyLoss()(logits_l1[is_labeled], batch.y1[is_labeled])
            is_labeled = batch.y2 > 0
            loss2 = nn.CrossEntropyLoss()(logits_l2[is_labeled], batch.y2[is_labeled])
            is_labeled = batch.y3 > 0
            loss3 = nn.CrossEntropyLoss()(logits_l3[is_labeled], batch.y3[is_labeled])
            loss += (loss1 + loss2 + loss3).item()

            yp_l1.append(torch.argmax(logits_l1, dim=-1))
            yp_l2.append(torch.argmax(logits_l2, dim=-1))
            yp_l3.append(torch.argmax(logits_l3, dim=-1))
            yt_l1.append(batch.y1)
            yt_l2.append(batch.y2)
            yt_l3.append(batch.y3)

        loss /= len(data_loader)
        yp_l1 = torch.cat(yp_l1, -1)
        yp_l2 = torch.cat(yp_l2, -1)
        yp_l3 = torch.cat(yp_l3, -1)
        yt_l1 = torch.cat(yt_l1, -1)
        yt_l2 = torch.cat(yt_l2, -1)
        yt_l3 = torch.cat(yt_l3, -1)

        return loss, yp_l1, yp_l2, yp_l3, yt_l1, yt_l2, yt_l3


def cross_validate(args):
    dataset = DataSet(args.batch_size)
    args.node_dim = dataset.node_dim
    args.edge_dim = dataset.edge_dim
    args.num_class_l1 = dataset.num_class_l1
    args.num_class_l2 = dataset.num_class_l2
    args.num_class_l3 = dataset.num_class_l3

    result = {
        'tier1': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        },
        'tier2': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        },
        'tier3': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        },
    }

    for __ in tqdm(range(100), unit_scale=True, desc='Running experiments...'):
        classifier = HierarchicalClassifier(args)
        classifier.train(dataset.train_loader, dataset.val_loader, dataset.weight)
        loss, yp_l1, yp_l2, yp_l3, yt_l1, yt_l2, yt_l3 = classifier.predict(dataset.test_loader)
        yp_l1 = yp_l1[yt_l1 > 0].cpu().numpy()
        yp_l2 = yp_l2[yt_l2 > 0].cpu().numpy()
        yp_l3 = yp_l3[yt_l3 > 0].cpu().numpy()
        yt_l1 = yt_l1[yt_l1 > 0].cpu().numpy()
        yt_l2 = yt_l2[yt_l2 > 0].cpu().numpy()
        yt_l3 = yt_l3[yt_l3 > 0].cpu().numpy()

        for measure in ['micro', 'macro', 'weighted']:
            result[f'tier1']['f1'][measure]['data'].append(f1_score(yp_l1, yt_l1, average=measure, zero_division=0))
            result[f'tier2']['f1'][measure]['data'].append(f1_score(yp_l2, yt_l2, average=measure, zero_division=0))
            result[f'tier3']['f1'][measure]['data'].append(f1_score(yp_l3, yt_l3, average=measure, zero_division=0))
            result[f'tier1']['precision'][measure]['data'].append(precision_score(yp_l1, yt_l1, average=measure, zero_division=0))
            result[f'tier2']['precision'][measure]['data'].append(precision_score(yp_l2, yt_l2, average=measure, zero_division=0))
            result[f'tier3']['precision'][measure]['data'].append(precision_score(yp_l3, yt_l3, average=measure, zero_division=0))
            result[f'tier1']['recall'][measure]['data'].append(recall_score(yp_l1, yt_l1, average=measure, zero_division=0))
            result[f'tier2']['recall'][measure]['data'].append(recall_score(yp_l2, yt_l2, average=measure, zero_division=0))
            result[f'tier3']['recall'][measure]['data'].append(recall_score(yp_l3, yt_l3, average=measure, zero_division=0))

        torch.cuda.empty_cache()
        dataset.shuffle()

    for measure in ['micro', 'macro', 'weighted']:
        result[f'tier1']['f1'][measure]['mean'] = np.mean(result[f'tier1']['f1'][measure]['data'])
        result[f'tier2']['f1'][measure]['mean'] = np.mean(result[f'tier2']['f1'][measure]['data'])
        result[f'tier3']['f1'][measure]['mean'] = np.mean(result[f'tier3']['f1'][measure]['data'])
        result[f'tier1']['f1'][measure]['std'] = np.std(result[f'tier1']['f1'][measure]['data'])
        result[f'tier2']['f1'][measure]['std'] = np.std(result[f'tier2']['f1'][measure]['data'])
        result[f'tier3']['f1'][measure]['std'] = np.std(result[f'tier3']['f1'][measure]['data'])

        result[f'tier1']['precision'][measure]['mean'] = np.mean(result[f'tier1']['precision'][measure]['data'])
        result[f'tier2']['precision'][measure]['mean'] = np.mean(result[f'tier2']['precision'][measure]['data'])
        result[f'tier3']['precision'][measure]['mean'] = np.mean(result[f'tier3']['precision'][measure]['data'])
        result[f'tier1']['precision'][measure]['std'] = np.std(result[f'tier1']['precision'][measure]['data'])
        result[f'tier2']['precision'][measure]['std'] = np.std(result[f'tier2']['precision'][measure]['data'])
        result[f'tier3']['precision'][measure]['std'] = np.std(result[f'tier3']['precision'][measure]['data'])

        result[f'tier1']['recall'][measure]['mean'] = np.mean(result[f'tier1']['recall'][measure]['data'])
        result[f'tier2']['recall'][measure]['mean'] = np.mean(result[f'tier2']['recall'][measure]['data'])
        result[f'tier3']['recall'][measure]['mean'] = np.mean(result[f'tier3']['recall'][measure]['data'])
        result[f'tier1']['recall'][measure]['std'] = np.std(result[f'tier1']['recall'][measure]['data'])
        result[f'tier2']['recall'][measure]['std'] = np.std(result[f'tier2']['recall'][measure]['data'])
        result[f'tier3']['recall'][measure]['std'] = np.std(result[f'tier3']['recall'][measure]['data'])

    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    with open(f'logs/{hash(str(args))}.json', 'w') as f:
        json.dump({**args.__dict__, **result}, f, indent=2)


def search(args):
    grid = [[64, 128, 256], [1, 2, 3]]
    for c in product(*grid):
        args.hid_dim = c[0]
        args.num_layers = c[1]
        cross_validate(args)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='gin', choices=['gcn', 'gat', 'gin', 'sage'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--node_dim', type=int)
    parser.add_argument('--edge_dim', type=int)
    parser.add_argument('--num_class_l1', type=int)
    parser.add_argument('--num_class_l2', type=int)
    parser.add_argument('--num_class_l3', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    cross_validate(args)

