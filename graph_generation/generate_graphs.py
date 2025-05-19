import torch
import random
import numpy as np
from typing import List
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset


class NormalizeLabels(T.BaseTransform):
    def __call__(self, data):
        if data.y is not None:
            norm = torch.norm(data.y.float().view(-1), p=2, keepdim=True)
            norm = norm + 1e-6
            data.y = data.y / norm
            data.y_norm = norm
        return data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, task_type='regression'):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
        self.task_type = task_type
        if task_type == 'regression':
            self.fc = torch.nn.Linear(out_channels, 1)
        else:  # classification
            self.fc = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        out = self.fc(x)
        if self.task_type == 'classification':
            out = out.argmax(dim=1)
        return out


class CustomGraphDataset(Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


# See: https://github.com/HongtengXu/SGWB-Graphon/blob/master/methods/simulator.py#L17
def synthesize_graphon(r: int = 1000, type_idx: int = 0) -> np.ndarray:
    """
        Synthesize graphons
        :param r: the resolution of discretized graphon
        :param type_idx: the type of graphon
        :return:
            w: (r, r) float array, whose element is in the range [0, 1]
    """
    u = ((np.arange(0, r) + 1) / r).reshape(-1, 1)  # (r, 1)
    v = ((np.arange(0, r) + 1) / r).reshape(1, -1)  # (1, r)

    if type_idx == 0:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v
    elif type_idx == 1:
        w = np.exp(-(u ** 0.7 + v ** 0.7))
    elif type_idx == 2:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.25 * (u ** 2 + v ** 2 + u ** 0.5 + v ** 0.5)
    elif type_idx == 3:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.5 * (u + v)
    elif type_idx == 4:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-2 * (u ** 2 + v ** 2)))
    elif type_idx == 5:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-(np.maximum(u, v) ** 2 + np.minimum(u, v) ** 4)))
    elif type_idx == 6:
        w = np.exp(-np.maximum(u, v) ** 0.75)
    elif type_idx == 7:
        w = np.exp(-0.5 * (np.minimum(u, v) + u ** 0.5 + v ** 0.5))
    elif type_idx == 8:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = np.log(1 + 0.5 * np.maximum(u, v))
    elif type_idx == 9:
        w = np.abs(u - v)
    elif type_idx == 10:
        w = 1 - np.abs(u - v)
    elif type_idx == 11:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), 0.8 * np.ones((r2, r2)))
    elif type_idx == 12:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), np.ones((r2, r2)))
        w = 0.8 * (1 - w)
    else:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v

    return w


def visualize_graphon(graphon: np.ndarray, save_path: str,
                      title: str = None, with_bar: bool = False):
    fig, ax = plt.subplots()
    plt.imshow(graphon, cmap='plasma', vmin=0.0, vmax=1.0)
    if with_bar:
        plt.colorbar()
    if title is not None:
        ax.set_title(title, fontsize=36)
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def simulate_graphs(device, w, num_graphs: int = 10, num_nodes: int = 200, graph_size: str = 'fixed',
                    feature_dim: int = 40, task_type='regression', normalizer_flag=False, balance_flag=False) -> List[Data]:
    """
        Simulate graphs with node features and continuous labels based on a graphon.

        :param w: a (r, r) discretized graphon
        :param num_graphs: the number of simulated graphs
        :param num_nodes: the number of nodes per graph
        :param graph_size: fix each graph size as num_nodes or sample the size randomly as num_nodes * (0.5 + uniform)
        :param feature_dim: the dimension of node features
        :return:
            graphs: a list of torch_geometric.data.Data objects
    """
    gcn = GCN(feature_dim, feature_dim // 2, feature_dim // 4, task_type=task_type).to(device)
    num_classes = int(feature_dim // 4)
    for param in gcn.parameters():
        if param.dim() > 1:
            # nn.init.kaiming_normal_(param)
            # nn.init.xavier_uniform_(param)
            nn.init.normal_(param, 0.0, 1.0)
    # feat_transform = T.NormalizeFeatures()
    # edge_transform = T.GCNNorm(add_self_loops=False)
    normalizer = NormalizeLabels()
    graphs = []
    r = w.shape[0]
    if graph_size == 'fixed':
        numbers = [num_nodes for _ in range(num_graphs)]
    elif graph_size == 'random':
        numbers = [int(num_nodes * (0.5 + np.random.rand())) for _ in range(num_graphs)]
    else:
        numbers = [num_nodes for _ in range(num_graphs)]

    for n in range(num_graphs):
        node_locs = (r * np.random.rand(numbers[n])).astype('int')
        graph = w[node_locs, :]
        graph = graph[:, node_locs]
        noise = np.random.rand(graph.shape[0], graph.shape[1])
        graph -= noise
        adj_matrix = (graph > 0).astype('float')

        edge_index = torch.tensor(np.vstack(adj_matrix.nonzero()), dtype=torch.long).to(device)
        features = torch.tensor(np.random.rand(len(adj_matrix), feature_dim), dtype=torch.float).to(device)
        # property_values = features.sum(dim=1).unsqueeze(1)
        data = Data(x=features, edge_index=edge_index)
        # data = feat_transform(data)
        # data = edge_transform(data)
        with torch.no_grad():
            if task_type == 'classification':
                if balance_flag:
                    pseudo_labels = gcn(data.x, data.edge_index, data.edge_weight)
                    total_samples = len(pseudo_labels)
                    desired_count_per_class = total_samples // num_classes
                    extra_samples = total_samples % num_classes
                    unique, counts = pseudo_labels.unique(return_counts=True)
                    print(f"Class distribution before balancing: {dict(zip(unique.tolist(), counts.tolist()))}")
                    device = pseudo_labels.device
                    all_classes = torch.arange(num_classes, device=device)
                    missing_classes = all_classes[~torch.isin(all_classes, unique)]
                    if len(missing_classes) > 0:
                        print(f"Missing classes: {missing_classes.tolist()}")
                        # available_indices = torch.arange(len(pseudo_labels), device=pseudo_labels.device)
                        # replace_indices = available_indices[torch.randperm(len(available_indices))[:len(missing_classes)]]
                        # pseudo_labels[replace_indices] = missing_classes
                        most_populated_class = unique[counts.argmax().item()]
                        most_populated_indices = (pseudo_labels == most_populated_class).nonzero(as_tuple=True)[0]
                        replace_indices = most_populated_indices[
                            torch.randperm(len(most_populated_indices))[:len(missing_classes)]]
                        pseudo_labels[replace_indices] = missing_classes

                    unique, counts = pseudo_labels.unique(return_counts=True)
                    class_to_indices = {cls.item(): (pseudo_labels == cls).nonzero(as_tuple=True)[0] for cls in unique}

                    unassigned_indices = []
                    for cls, indices in class_to_indices.items():
                        if len(indices) > desired_count_per_class:
                            excess_indices = indices[
                                torch.randperm(len(indices))[:(len(indices) - desired_count_per_class)]]
                            unassigned_indices.extend(excess_indices.tolist())
                        elif len(indices) < desired_count_per_class:
                            deficit = desired_count_per_class - len(indices)
                            if len(unassigned_indices) >= deficit:
                                new_indices = torch.tensor(unassigned_indices[:deficit], device=device)
                                pseudo_labels[new_indices] = cls
                                unassigned_indices = unassigned_indices[deficit:]
                            else:
                                surplus_classes = {k: v for k, v in class_to_indices.items() if
                                                   len(v) > desired_count_per_class}
                                surplus_indices = torch.cat([v for k, v in surplus_classes.items()])
                                additional_indices = surplus_indices[torch.randperm(len(surplus_indices))[:deficit]]
                                pseudo_labels[additional_indices] = cls

                    if extra_samples > 0:
                        # Calculate the actual number of samples for each class
                        unique, counts = pseudo_labels.unique(return_counts=True)
                        class_to_count = dict(zip(unique.tolist(), counts.tolist()))

                        # Initialize deficits and surpluses for each class
                        # Deficits: Classes with fewer samples than the desired count
                        # Surpluses: Classes with more samples than the desired count
                        deficits = {cls: max(0, desired_count_per_class - class_to_count.get(cls, 0)) for cls in
                                    range(num_classes)}
                        surpluses = {cls: max(0, class_to_count.get(cls, 0) - desired_count_per_class) for cls in
                                     range(num_classes)}

                        # Identify classes with deficits and surpluses
                        deficit_classes = [cls for cls, deficit in deficits.items() if deficit > 0]
                        surplus_classes = [cls for cls, surplus in surpluses.items() if surplus > 0]

                        # Supplement the classes with deficits
                        for deficit_cls in deficit_classes:
                            # Determine the deficit for the current class
                            deficit = deficits[deficit_cls]
                            while deficit > 0 and surplus_classes:
                                # Find the class with the highest surplus
                                surplus_cls = max(surplus_classes, key=lambda cls: surpluses[cls])

                                # Get the indices of samples from the surplus class
                                surplus_indices = (pseudo_labels == surplus_cls).nonzero(as_tuple=True)[0]

                                # Determine the number of samples to transfer
                                transfer_count = min(deficit, len(surplus_indices))
                                # Randomly select indices from the surplus class
                                chosen_indices = surplus_indices[torch.randperm(len(surplus_indices))[:transfer_count]]

                                # Update pseudo labels to assign these samples to the deficit class
                                pseudo_labels[chosen_indices] = deficit_cls
                                # Decrease the deficit for the current class
                                deficit -= transfer_count
                                # Decrease the surplus for the surplus class
                                surpluses[surplus_cls] -= transfer_count

                                # If the surplus class runs out of excess samples, remove it from the list
                                if surpluses[surplus_cls] == 0:
                                    surplus_classes.remove(surplus_cls)

                    unique, counts = pseudo_labels.unique(return_counts=True)
                    print(f"After balancing: {dict(zip(unique.tolist(), counts.tolist()))}")
                    data.y = pseudo_labels
                else:
                    pseudo_labels = gcn(data.x, data.edge_index, data.edge_weight)
                    unique, counts = pseudo_labels.unique(return_counts=True)
                    print(f"unique and counts: {dict(zip(unique.tolist(), counts.tolist()))}")
                    data.y = pseudo_labels
            else:
                property_values = gcn(data.x, data.edge_index, data.edge_weight)
                data.y = property_values
        if normalizer_flag:
            data = normalizer(data)
        graphs.append(data)

    return graphs


# seed python, numpy, pytorch
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_everything(42)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    graph_result_dir = 'graph_generation/graphon_results'

    graphon = synthesize_graphon(r=1000, type_idx=0)
    # visualize_graphon(graphon, save_path=os.path.join(graph_result_dir, 'graphon_0.pdf'))
    graphs = simulate_graphs(device, graphon, num_graphs=50000, num_nodes=100,
                             graph_size='random', task_type='regression', normalizer_flag=True)
    dataset = CustomGraphDataset(graphs)
    torch.save(dataset, f'graph_generation/generated_graph_datasets/dataset_0.pt')

    graphon = synthesize_graphon(r=1000, type_idx=0)
    graphs = simulate_graphs(device, graphon, num_graphs=50000, num_nodes=100,
                             graph_size='random', task_type='classification')
    dataset = CustomGraphDataset(graphs)
    torch.save(dataset, f'graph_generation/generated_graph_datasets/cls_dataset_0.pt')
