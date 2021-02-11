import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class GIN(MessagePassing):
    def __init__(self, emb_dim):
        super(GIN, self).__init__()
        # self.mlp = torch.nn.Sequential(
        #     nn.Linear(2 * emb_dim, 2 * emb_dim),
        #     nn.BatchNorm1d(2 * emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(2 * emb_dim, emb_dim))
        self.mlp = torch.nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr
        # return torch.cat([x_j, edge_attr], dim=-1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_class, num_layers, JK='sum'):
        super(GNN, self).__init__()
        self.drop = torch.nn.Dropout(p=0.05)
        self.num_class = num_class
        self.num_layers = num_layers
        self.linear_node = nn.Linear(node_dim, hid_dim)
        self.linear_edge = nn.Linear(edge_dim, hid_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hid_dim, 2 * hid_dim),
            nn.BatchNorm1d(2 * hid_dim),
            nn.PReLU(),
            nn.Linear(2 * hid_dim, num_class)
        )
        self.JK = JK

        self.gnns = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.gnns.append(GIN(hid_dim))

    def forward(self, x, edge_index, e):
        h = self.linear_node(x)
        e = self.linear_edge(e)
        h_list = [h]
        for layer in range(self.num_layers):
            h = self.gnns[layer](h_list[layer], edge_index, e)
            h = self.drop(F.leaky_relu(h, negative_slope=0.2))
            h_list.append(h)

        if self.JK == "last":
            h = h_list[-1]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            h = torch.sum(torch.cat(h_list), 0)
        h = self.output_layer(h)
        return h


class Ensemble(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hid_dim, num_class, num_layers, JK='sum'):
        super(Ensemble, self).__init__()
        self.gnn1 = GNN(node_dim, edge_dim, hid_dim, num_class, num_layers, JK=JK)
        self.gnn2 = GNN(node_dim, edge_dim, hid_dim, num_class, num_layers, JK=JK)
        self.gnn3 = GNN(node_dim, edge_dim, hid_dim, num_class, num_layers, JK=JK)

    def forward(self, x, edge_index, e):
        return self.gnn1(x, edge_index, e) + self.gnn2(x, edge_index, e) + self.gnn3(x, edge_index, e)


if __name__ == "__main__":
    pass

