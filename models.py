import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_sparse import SparseTensor, cat, matmul
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

# from memory_profiler import profile


class GrantConv(nn.Module):
    def __init__(self, device: torch.device, sparse: bool = False):
        super(GrantConv, self).__init__()
        self.device = device
        self.sparse = sparse

    def forward(self, k, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        idx = torch.arange(num_nodes)
        with torch.no_grad():
            if self.sparse:  # sparse matrix
                I = SparseTensor(row=idx, col=idx,
                                 value=torch.ones(num_nodes), sparse_sizes=(num_nodes, num_nodes))
                A = SparseTensor(row=edge_index[0], col=edge_index[1],
                                 value=edge_weight, sparse_sizes=(num_nodes, num_nodes))
                M = I.clone().to(self.device)
                novel_A = A.clone()
                for _ in range(1, k):
                    if _ > 1:
                        novel_A = matmul(novel_A, A.clone())
                    M = cat((M, novel_A), 1)
            else:  # dense matrix
                I = torch.eye(num_nodes)
                adj = torch.zeros((num_nodes, num_nodes), dtype=x.dtype)
                for i in range(edge_index.size(1)):
                    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                    adj[src, dst] = edge_weight[i].item()
                    adj[dst, src] += edge_weight[i].item()
                M = I.to(self.device)
                adj = adj.to(self.device)
                for _ in range(1, k):
                    if _ > 1:
                        novel_A = torch.matmul(novel_A, adj)
                    else:
                        novel_A = adj
                    M = torch.cat((M, novel_A), dim=1)

        blocks = [x for _ in range(k)]
        N = torch.block_diag(*blocks).to(self.device)
        # Get sparse_N
        # row, col = N.nonzero(as_tuple=True)
        # values = N[row, col]
        # sparse_N = SparseTensor(row=row, col=col, value=values, sparse_sizes=N.shape)

        # M: n * kn (no grad)
        # N: kn * Fk (with grad)
        # C: n * Fk (with grad)
        M.requires_grad_(False)
        if self.sparse:  # For sparse matrix
            C = matmul(M, N)
        else:
            C = torch.matmul(M.clone().detach(), N)

        return C


class GrantGCN(nn.Module):
    # mlp_dims: [[a, b], [b, c]...]
    # last_linear_dims: tuple, for the last linear (in_dim, out_dim)
    def __init__(self, n_layers, x_dim, mlp_dims, last_linear_dims,
                 task_type, sparse, device, atom_enc_flag, dataset_name):
        in_dim, out_dim = last_linear_dims
        super(GrantGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # ogbg-molhiv
        self.atom_enc_flag = atom_enc_flag
        if self.atom_enc_flag:
            self.atom_encoder = AtomEncoder(x_dim)

        for i in range(n_layers):
            self.convs.append(GrantConv(device, sparse))
            if n_layers == 1 or i < n_layers - 1:  # example: n_layers = 1 or n_layers >= 2
                # XW + RELU
                self.convs.append(nn.Linear(mlp_dims[i][0], mlp_dims[i][1]))
                if self.atom_enc_flag:
                    self.batch_norms.append(nn.BatchNorm1d(mlp_dims[i][1]))
                if task_type in ('node_classification', 'graph_classification'):
                    self.convs.append(nn.ReLU())
                elif task_type in ('node_regression', 'graph_regression'):
                    self.convs.append(nn.ReLU())

        # If only 1 layer, the in_dim should not be "k_list[-1] * mlp_dims[-1][1]"
        if n_layers == 1:
            in_dim = mlp_dims[0][1]

        if task_type == 'node_regression':
            self.convs.append(nn.Linear(in_dim, out_dim))
            self.convs.append(nn.Linear(out_dim, 1))  # the node-level property prediction
        elif task_type == 'graph_regression':
            self.convs.append(nn.Linear(in_dim, out_dim))
            self.convs.append(nn.Linear(out_dim, 1))  # the graph-level property prediction
        elif task_type == 'node_classification':
            self.convs.append(nn.Linear(in_dim, out_dim))
        elif task_type == 'graph_classification' and dataset_name == 'ogbg-molhiv':
            self.convs.append(nn.Linear(in_dim, out_dim))
            self.convs.append(nn.Linear(out_dim, 1))
        elif task_type == 'graph_classification' and dataset_name == 'ogbg-molpcba':
            # NOTE: The draft model outputs the dimension as 1, while the `target_y` dimension is 128.
            # To align dimensions for loss calculation during training, We fixed it in the current implementation.
            self.convs.append(nn.Linear(in_dim, out_dim))
            self.convs.append(nn.Linear(out_dim, 128))

        self.task_type = task_type
        self.n_layers = n_layers

    def forward(self, x, k_list, edge_data, batch: Optional[torch.Tensor] = None):
        if self.atom_enc_flag:
            x = self.atom_encoder(x.long())

        edge_index, edge_weight = edge_data
        cnt = 0
        conv_lens = len(self.convs)
        end_len = conv_lens - 1 if self.task_type == 'node_classification' else conv_lens - 2
        for i in range(0, end_len):
            conv = self.convs[i]
            if isinstance(conv, GrantConv):
                gc_out = conv(k_list[cnt], x, edge_index, edge_weight)
                cnt += 1
            elif isinstance(conv, nn.Linear):
                linear_out = conv(gc_out)
                if self.atom_enc_flag and len(self.batch_norms) >= cnt:
                    linear_out = self.batch_norms[cnt - 1](linear_out)
            elif isinstance(conv, nn.ReLU):
                x = conv(linear_out)

        if self.n_layers == 1:
            in_x = x
        else:
            in_x = gc_out

        # Refer to: https://github.com/jz48/RegExplainer/blob/main/gnns/DenseRegGCN_reg.py#L7
        if self.task_type == 'node_regression':
            out = self.convs[-2](in_x)
            out = self.convs[-1](out)
        # Refer to: https://github.com/jz48/RegExplainer/blob/main/gnns/GraphGCN_reg.py#L9
        elif self.task_type == 'graph_regression':
            if batch is None:
                batch = torch.zeros(in_x.size(0), dtype=torch.long)
            linear_in = global_mean_pool(in_x, batch)  # graph representation
            out = self.convs[-2](linear_in)
            out = self.convs[-1](out)
        elif self.task_type == 'node_classification':
            out = self.convs[-1](in_x)
        elif self.task_type == 'graph_classification':
            if batch is None:
                batch = torch.zeros(in_x.size(0), dtype=torch.long)
            linear_in = global_mean_pool(in_x, batch)
            out = self.convs[-2](linear_in)
            out = self.convs[-1](out)
        return out

    def mse_loss(self, y_pred, y, reduction='mean'):
        return F.mse_loss(y_pred, y, reduction=reduction)

    def mae(self, y_pred, y):
        return F.l1_loss(y_pred, y)
        # return torch.mean(torch.abs(y - y_pred))


class SimpleGrantGCN(nn.Module):
    def __init__(self, n_layers, mlp_dims, last_linear_dims, task_type):
        assert task_type in ('node_regression', 'node_classification')
        super(SimpleGrantGCN, self).__init__()
        in_dim, out_dim = last_linear_dims
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(nn.Linear(mlp_dims[i][0], mlp_dims[i][1]))
            self.convs.append(nn.ReLU())
        self.convs.append(nn.Linear(in_dim, out_dim))
        if task_type == 'node_regression':
            self.convs.append(nn.Linear(out_dim, 1))

    def forward(self, x):
        for i in range(0, len(self.convs)):
            conv = self.convs[i]
            x = conv(x)
        return x

    def mse_loss(self, y_pred, y, reduction='mean'):
        return F.mse_loss(y_pred, y, reduction=reduction)

    def mae(self, y_pred, y):
        return F.l1_loss(y_pred, y)
