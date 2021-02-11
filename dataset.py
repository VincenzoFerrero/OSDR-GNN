import torch
import networkx as nx
import torch.nn.functional as F
from collections import Counter
from autodesk_colab_KG import load_data
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split


def get_vocab():
    graphs = load_data('data/autodesk_colab_fullv3_202010291746.csv')
    subfunc_counts, tier1_counts, tier2_counts, tier3_counts = [], [], [], []
    vocab = {
        'component_basis': set(),
        'sys_name': set(),
        'sys_type_name': set(),
        'material_name': set(),
        'subfunction_basis': set(),
        'tier_1_function': set(),
        'tier_2_function': set(),
        'tier_3_function': set(),
        'flow': set(),
    }
    for g in graphs:
        for __, attr in g.nodes(data=True):
            vocab['component_basis'].add(attr['component_basis'])
            vocab['sys_name'].add(attr['sys_name'])
            vocab['sys_type_name'].add(attr['sys_type_name'])
            vocab['material_name'].add(attr['material_name'])
            vocab['subfunction_basis'].add(attr['subfunction_basis'])
            vocab['tier_1_function'].add(attr['tier_1_function'])
            vocab['tier_2_function'].add(attr['tier_2_function'])
            vocab['tier_3_function'].add(attr['tier_3_function'])
            subfunc_counts.append(attr['subfunction_basis'])
            tier1_counts.append(attr['tier_1_function'])
            tier2_counts.append(attr['tier_2_function'])
            tier3_counts.append(attr['tier_3_function'])
        for attr in g.edges(data=True):
            if 'input_flow' in attr[-1]:
                vocab['flow'].add(attr[-1]['input_flow'])
            if 'output_flow' in attr[-1]:
                vocab['flow'].add(attr[-1]['output_flow'])
    for k, v in vocab.items():
        vocab[k] = {s: idx for idx, s in enumerate(sorted(v))}

    subfunc_counts = [vocab['subfunction_basis'][l] for l in subfunc_counts]
    tier1_counts = [vocab['tier_1_function'][l] for l in tier1_counts]
    tier2_counts = [vocab['tier_2_function'][l] for l in tier2_counts]
    tier3_counts = [vocab['tier_3_function'][l] for l in tier3_counts]

    subfunc_w = [0.] * len(vocab['subfunction_basis'])
    tier1_w = [0.] * len(vocab['tier_1_function'])
    tier2_w = [0.] * len(vocab['tier_2_function'])
    tier3_w = [0.] * len(vocab['tier_3_function'])

    for k, v in Counter(subfunc_counts).items():
        subfunc_w[k] = len(subfunc_counts) / v
    for k, v in Counter(tier1_counts).items():
        tier1_w[k] = len(tier1_counts) / v
    for k, v in Counter(tier2_counts).items():
        tier2_w[k] = len(tier2_counts) / v
    for k, v in Counter(tier3_counts).items():
        tier3_w[k] = len(tier3_counts) / v

    weights = {0: torch.tensor(subfunc_w), 1: torch.tensor(tier1_w),
               2: torch.tensor(tier2_w), 3: torch.tensor(tier3_w)}
    return graphs, vocab, weights


def preprocess():
    graphs = []
    data, vocab, weights = get_vocab()

    for graph in data:
        if graph.number_of_nodes() < 3 or graph.number_of_edges() < 2:
            continue

        mappings = {n: idx for idx, n in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mappings)

        nodes = torch.cat((
            F.one_hot(
                torch.tensor([vocab['component_basis'][n[-1]['component_basis']] for n in graph.nodes(data=True)]),
                len(vocab['component_basis'])),
            F.one_hot(
                torch.tensor([vocab['sys_name'][n[-1]['sys_name']] for n in graph.nodes(data=True)]),
                len(vocab['sys_name'])),
            F.one_hot(
                torch.tensor([vocab['sys_type_name'][n[-1]['sys_type_name']] for n in graph.nodes(data=True)]),
                len(vocab['sys_type_name'])),
            F.one_hot(
                torch.tensor([vocab['material_name'][n[-1]['material_name']] for n in graph.nodes(data=True)]),
                len(vocab['material_name'])),
        ), -1)

        y = torch.tensor([vocab[f'subfunction_basis'][n[-1][f'subfunction_basis']] for n in graph.nodes(data=True)])
        y1 = torch.tensor([vocab[f'tier_1_function'][n[-1][f'tier_1_function']] for n in graph.nodes(data=True)])
        y2 = torch.tensor([vocab[f'tier_2_function'][n[-1][f'tier_2_function']] for n in graph.nodes(data=True)])
        y3 = torch.tensor([vocab[f'tier_3_function'][n[-1][f'tier_3_function']] for n in graph.nodes(data=True)])

        edges = []
        for edge in graph.edges(data=True):
            edges.append(torch.zeros(2 * len(vocab['flow']) + 1))
            if len(edge[-1]) == 0:
                edges[-1][0] = 1.
            else:
                if 'input_flow' in edge[-1]:
                    edges[-1][vocab['flow'][edge[-1]['input_flow']] + 1] = 1.
                if 'output_flow' in edge[-1]:
                    edges[-1][vocab['flow'][edge[-1]['output_flow']] + len(vocab['flow']) + 1] = 1.

        edges = torch.stack(edges)
        edge_index = torch.tensor([[e[0], e[1]] for e in graph.edges()]).transpose(1, 0)
        graphs.append(Data(x=nodes, edge_index=edge_index, e=edges, y=y, y1=y1, y2=y2, y3=y3))

    return graphs, vocab, weights


class DataSet(object):
    def __init__(self, batch_size):
        self.graphs, self.vocab, self.weight = preprocess()
        self.batch_size = batch_size
        self.node_dim = self.graphs[0].x.shape[-1]
        self.edge_dim = self.graphs[0].e.shape[-1]
        self.num_class_l1 = len(self.vocab['tier_1_function'])
        self.num_class_l2 = len(self.vocab['tier_2_function'])
        self.num_class_l3 = len(self.vocab['tier_3_function'])
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.shuffle()

    def shuffle(self):
        train, test = train_test_split(self.graphs, test_size=.4, shuffle=True)
        train, val = train_test_split(self.graphs, test_size=.1, shuffle=True)
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    graphs, vocab, w = preprocess(1)
    print('heer')
