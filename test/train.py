import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
from args import args
from gnn import Ensemble
from dataset import DataSet, preprocess, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report


seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class Model(object):
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_layers = args.num_layers
        self.num_class = args.num_class
        self.patience = args.patience
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.hid_dim = args.hid_dim
        self.tier = args.tier
        self.lr = args.lr
        self.gnn = Ensemble(self.node_dim, self.edge_dim, self.hid_dim, self.num_class, self.num_layers).to(self.device)

    def load(self):
        if os.path.exists('checkpoint.pkl'):
            self.gnn.load_state_dict(torch.load('checkpoint.pkl'))
        else:
            raise Exception('Checkpoint not found ...')

    def train(self, train_loader, val_loader, weights):
        best_loss, best_acc, best_state, patience_count = 1e9, .0, None, 0
        optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        weights = weights.to(self.device)

        for epoch in range(self.num_epochs):
            self.gnn.train()
            start = time.time()
            epoch_loss, epoch_acc = 0., 0.
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                logits = self.gnn(batch.x.float(), batch.edge_index, batch.e.float())
                is_labeled = batch.y > 0
                loss = nn.CrossEntropyLoss(weight=weights)(logits[is_labeled], batch.y[is_labeled])
                yp = torch.argmax(logits, dim=-1)
                epoch_acc += (yp == batch.y).sum().item() / batch.y.shape[0]
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            scheduler.step()
            end = time.time()
            val_loss, val_acc = self.eval(val_loader)
            print(f'Epoch: {epoch + 1:03d}/{self.num_epochs}, Time: {end-start:.2f}s, '
                  f'Train [Loss: {epoch_loss / len(train_loader):.4f}, '
                  f'Accuracy: {epoch_acc / len(train_loader):.2f}%], Val [Loss: {val_loss: .4f}, '
                  f'Accuracy: {val_acc:.2f}%]')

            if best_loss > val_loss:
                best_loss = val_loss
                best_state = self.gnn.state_dict()
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == self.patience:
                print('Early stopping ...')
                break

        self.gnn.load_state_dict(best_state)
        torch.save(best_state, 'checkpoint.pkl')

    @torch.no_grad()
    def eval(self, data_loader):
        self.gnn.eval()
        loss, ac = 0., 0.
        for batch in data_loader:
            batch = batch.to(self.device)
            logits = self.gnn(batch.x.float(), batch.edge_index, batch.e.float())
            is_labeled = batch.y > 0
            loss = nn.CrossEntropyLoss()(logits[is_labeled], batch.y[is_labeled]).item()
            yp = torch.argmax(logits, dim=-1)
            ac += (yp == batch.y).sum().item() / batch.y.shape[0]
        return loss / len(data_loader), ac / len(data_loader)

    @torch.no_grad()
    def predict(self, data_loader):
        self.load()
        self.gnn.eval()
        yp, yt = [], []
        for batch in data_loader:
            batch = batch.to(self.device)
            logits = self.gnn(batch.x.float(), batch.edge_index, batch.e.float())
            yp.append(torch.argmax(logits, dim=-1))
            yt.append(batch.y)
        return torch.cat(yp, -1).cpu().numpy(), torch.cat(yt, -1).cpu().numpy()


def cross_validate(args):
    graphs, vocab, weight = preprocess(args.tier)
    args.node_dim = graphs[0].x.shape[-1]
    args.edge_dim = graphs[0].e.shape[-1]
    args.num_class = len(vocab[f'tier_{args.tier}_function'])
    kf = KFold(n_splits=10)

    for __ in range(1):
        for train_idx, test_idx in kf.split(graphs):
            start = time.time()
            classifier = Model(args)
            val_idx = train_idx[-10:]
            train_idx = train_idx[:-10]
            train_loader = DataLoader([graphs[idx] for idx in train_idx], batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader([graphs[idx] for idx in val_idx], batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader([graphs[idx] for idx in test_idx], batch_size=args.batch_size, shuffle=False)
            classifier.train(train_loader, val_loader, weight)
            yp_test, yt_test = classifier.predict(test_loader)
            is_labeled = yt_test > 0
            print(f'Test Accuracy: {accuracy_score(yt_test[is_labeled], yp_test[is_labeled]):.2f}%')
            print(classification_report(yt_test[is_labeled], yp_test[is_labeled]))


if __name__ == '__main__':
    args.tier = 3
    cross_validate(args)
    # dataset = DataSet(args.batch_size, args.tier)
    # args.node_dim = dataset.node_dim
    # args.edge_dim = dataset.edge_dim
    # args.num_class = dataset.num_tier[args.tier]
    #
    # model = Model(args)
    # model.train(dataset.train_loader, dataset.val_loader, dataset.weight)
    # yp_test, yt_test = model.predict(dataset.test_loader)
    # is_labeled = yt_test > 0
    # print(f'Test Accuracy: {accuracy_score(yt_test[is_labeled], yp_test[is_labeled]):.2f}%')
    # print(classification_report(yt_test[is_labeled], yp_test[is_labeled]))
