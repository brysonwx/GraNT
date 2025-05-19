import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import os.path as osp
from typing import Any
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import random_split
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import subgraph, k_hop_subgraph, to_undirected

# evaluator = Evaluator('ogbg-molhiv')
evaluator = Evaluator('ogbg-molpcba')


# seed python, numpy, pytorch
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_cls_dataset(path, name):
    assert name in ['ogbg-molhiv', 'ogbg-molpcba', 'gen-cls', ]

    root_path = osp.expanduser('~/datasets')
    if name.startswith('ogbg'):
        return PygGraphPropPredDataset(root=osp.join(root_path, 'OGB'), name=name)

    if name == 'gen-cls':
        dataset = torch.load('graph_generation/generated_graph_datasets/cls_dataset_0.pt')
        return dataset


def get_reg_dataset(path, name):
    assert name in ['QM9', 'ZINC', 'gen-reg', ]
    base_path = osp.expanduser('~/datasets')

    # graph regression
    if name == 'QM9':
        return QM9(root=osp.join(base_path, 'QM9'), transform=T.NormalizeFeatures())

    if name == 'ZINC':
        return ZINC(root=osp.join(base_path, 'ZINC-train'))

    # diverse graphs
    if name == 'gen-reg':
        dataset = torch.load('graph_generation/generated_graph_datasets/dataset_0.pt')
        return dataset


def generate_split(num_samples: int, train_ratio: float, val_ratio: float, ogb_split_idx: Any):
    if ogb_split_idx is not None:
        idx_train = ogb_split_idx['train']
        idx_val = ogb_split_idx['valid']
        idx_test = ogb_split_idx['test']
    else:
        train_len = int(num_samples * train_ratio)
        val_len = int(num_samples * val_ratio)
        test_len = num_samples - train_len - val_len
        train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))
        idx_train, idx_val, idx_test = train_set.indices, val_set.indices, test_set.indices

    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    return train_mask, val_mask, test_mask
    # return idx_train, idx_val, idx_test


def get_split_data(data, train_ratio, val_ratio, ogb_split_idx, transform, device):
    # 6:2:2
    train_mask, val_mask, test_mask = generate_split(
        data.num_nodes, train_ratio=train_ratio, val_ratio=val_ratio, ogb_split_idx=ogb_split_idx)
    train_node_indices = torch.nonzero(train_mask).squeeze().to(device)
    edge_index = data.edge_index
    if edge_index.device == torch.device('cpu'):
        edge_index = edge_index.to(device)
    train_edge_index, _ = subgraph(train_node_indices, edge_index, relabel_nodes=True)
    train_data = Data(
        x=data.x[train_node_indices],
        y=data.y[train_node_indices],
        edge_index=train_edge_index,
    )

    train_data = transform(train_data)

    val_node_indices = torch.nonzero(val_mask).squeeze().to(device)
    val_edge_index, _ = subgraph(val_node_indices, edge_index, relabel_nodes=True)
    val_data = Data(
        x=data.x[val_node_indices],
        y=data.y[val_node_indices],
        edge_index=val_edge_index,
    )

    val_data = transform(val_data)

    test_node_indices = torch.nonzero(test_mask).squeeze().to(device)
    test_edge_index, _ = subgraph(test_node_indices, edge_index, relabel_nodes=True)
    test_data = Data(
        x=data.x[test_node_indices],
        y=data.y[test_node_indices],
        edge_index=test_edge_index,
    )

    test_data = transform(test_data)

    return train_data, val_data, test_data


def modify_mlp_dims(sampling_type, mlp_dims, n_layers, k_list, x_dim):
    if sampling_type in (1, 2, 3, 4):
        len_mlp_dims = len(mlp_dims)
        if n_layers != 1:
            assert len_mlp_dims == (n_layers - 1)
        # override the first_dim in mlp_dims
        first_dim = k_list[0] * x_dim
        for i in range(len_mlp_dims):
            mlp_dims[i][0] = first_dim
            if i+1 < len_mlp_dims:
                first_dim = k_list[i+1] * mlp_dims[i][-1]
        in_dim = k_list[-1] * mlp_dims[-1][1]
    elif sampling_type == 5:
        # [[40, 20], [20, 10]]
        # [[384, 192], [192, 96]]
        new_dim = k_list[0] * x_dim
        mlp_dims[0][0] = new_dim
        in_dim = mlp_dims[-1][1]
    return in_dim


# See: https://github.com/chennnM/GCNII/blob/master/utils.py#L13
def accuracy(output, labels):  # node classification accuracy
    if output.dim() == 1:
        output = output.unsqueeze(1)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.unsqueeze(1).detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    rocauc_list = []

    # Ensure y_true is a numpy array
    y_true = y_true.detach().cpu().numpy()

    # Handle one-hot encoding for multi-class classification
    if y_true.ndim == 1:
        n_classes = y_pred.size(-1)
        y_true = label_binarize(y_true, classes=np.arange(n_classes))

    # Compute softmax probabilities
    y_pred = F.softmax(y_pred, dim=-1).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive and one negative label
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # Filter labeled data
            is_labeled = y_true[:, i] == y_true[:, i]  # For non-NaN data
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i], multi_class='ovr')
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


# For graph regression / node regression / graph classification / node classification
# sampling_type = 1, 2, 3, 4
def train_with_st_1234(model, optimizer, train_loader, k_list, transform,
                       target_index, total_len, clip_max, task_type, device, rocauc_flag=False):
    model.train()
    optimizer.zero_grad()
    total_mae = 0
    total_loss = 0
    total_acc = 0
    total_rocauc = 0
    total_ap = 0
    all_clf_labels = []
    all_clf_logits = []
    batch_cnt = len(train_loader)
    fail_batch_cnt = 0
    for data in tqdm(train_loader, desc='Processing train_loader batches'):
        data = transform(data)
        x = data.x.to(device)
        if x.dtype == torch.int64:
            x = x.float()
        edge_index = data.edge_index.to(device)
        edge_weight = data.edge_weight.to(device)
        edge_data = (edge_index, edge_weight)
        y = data.y.to(device)
        batch = data.batch.to(device)
        output = model(x, k_list, edge_data, batch)
        if target_index != -1:  # qm9, graph regression
            target_y = y[:, target_index].view_as(output)
        else:
            if task_type == 'node_classification':
                target_y = y.squeeze(-1)
                criterion = torch.nn.CrossEntropyLoss()
            elif task_type == 'graph_classification':  # ogbg-molhiv, ogbg-molpcba
                output = output.squeeze(-1)
                target_y = y.squeeze(-1).float()
                criterion = torch.nn.BCEWithLogitsLoss()
            else:  # e.g., zinc, graph regression, node regression
                target_y = y.view_as(output)

        if task_type == 'node_classification':
            loss = criterion(output, target_y)
            p_train = F.log_softmax(output, dim=1)
            if rocauc_flag:  # gen-cls
                rocauc = eval_rocauc(target_y, p_train)
                total_rocauc += rocauc
            else:
                # acc = eval_acc(target_y, p_train)
                acc = accuracy(output, target_y).to(device)
                total_acc += acc.item()
        elif task_type == 'graph_classification':  # ogbg-molhiv, ogbg-molpcba
            if not rocauc_flag:  # i.e., ogbg-molpcba
                # NOTE: The draft model outputs a tensor of shape [batch_size, 1],
                # while the `target_y` dimension is 128. To align dimensions for loss calculation,
                # we initially expanded the output (i.e., the commented code below):
                # `output_tmp = output.unsqueeze(-1).expand(-1, 128)`
                # This was later fixed in the current implementation.
                is_labeled = target_y == target_y
                # output_ls = output_tmp[is_labeled]
                output_ls = output[is_labeled]
                target_y_ls = target_y[is_labeled]
                loss = criterion(output_ls, target_y_ls)
            else:
                loss = criterion(output, target_y)
            p_train = torch.sigmoid(output)
            if p_train.dim() == 1 and target_y.dim() == 1:
                p_train = p_train.unsqueeze(1)
                target_y = target_y.unsqueeze(1)

            if not rocauc_flag:  # i.e., ogbg-molpcba
                # p_train = p_train.unsqueeze(-1).expand(-1, 128)
                ap_list_flag = False
                if torch is not None and isinstance(target_y, torch.Tensor):
                    target_y = target_y.detach().cpu().numpy()

                ap_list = []
                for i in range(target_y.shape[1]):
                    # AUC is only defined when there is at least one positive data.
                    if np.sum(target_y[:, i] == 1) > 0 and np.sum(target_y[:, i] == 0) > 0:
                        # # ignore nan values
                        # is_labeled = y_true[:,i] == y_true[:,i]
                        # ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])
                        ap_list.append(i)

                if len(ap_list) == 0:
                    fail_batch_cnt += 1
                    ap_list_flag = True

                if not ap_list_flag:
                    ap_batch = evaluator.eval({'y_pred': p_train, 'y_true': target_y})['ap']
                    total_ap += ap_batch
                # ap_batch = evaluator.eval({'y_pred': p_train, 'y_true': target_y})['ap']
                # total_ap += ap_batch
            else:
                all_clf_logits.append(p_train)
                all_clf_labels.append(target_y)
        else:
            loss = model.mse_loss(output, target_y)
            total_mae += model.mae(output, target_y).item()

        loss = loss / batch_cnt
        loss.backward()
        total_loss += loss.item() * data.num_graphs * batch_cnt
    # See: https://github.com/XieResearchGroup/Physics-aware-Multiplex-GNN/blob/main/main_qm9.py#L111
    clip_grad_norm_(model.parameters(), clip_max)
    optimizer.step()

    if task_type == 'node_classification':
        if rocauc_flag:
            return total_loss / total_len, total_rocauc / batch_cnt
        else:
            return total_loss / total_len, total_acc / batch_cnt
    elif task_type == 'graph_classification':
        if rocauc_flag:
            all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)
            rocauc = evaluator.eval({'y_pred': all_clf_logits, 'y_true': all_clf_labels})['rocauc']
            return total_loss / total_len, rocauc
        else:  # i.e., ogbg-molpcba
            return total_loss / total_len, total_ap / (batch_cnt - fail_batch_cnt)
    else:
        return total_loss / total_len, total_mae / total_len


# For graph regression / node regression / graph classification / node classification
# sampling_type = 1, 2, 3, 4
def validate_with_st_1234(model, val_loader, k_list, transform, target_index,
                          total_len, device, task_type, rocauc_flag=False):
    model.eval()
    with torch.no_grad():
        total_mae = 0
        total_loss = 0
        total_acc = 0
        total_rocauc = 0
        total_ap = 0
        all_clf_labels = []
        all_clf_logits = []
        batch_cnt = len(val_loader)
        for data in tqdm(val_loader, desc='Processing val_loader batches'):
            data = transform(data)
            x = data.x.to(device)
            if x.dtype == torch.int64:
                x = x.float()
            edge_index = data.edge_index.to(device)
            edge_weight = data.edge_weight.to(device)
            edge_data = (edge_index, edge_weight)
            y = data.y.to(device)
            batch = data.batch.to(device)
            output = model(x, k_list, edge_data, batch)
            if target_index != -1:
                target_y = y[:, target_index].view_as(output)
            else:
                if task_type == 'node_classification':
                    target_y = y.squeeze(-1)
                    criterion = torch.nn.CrossEntropyLoss()
                elif task_type == 'graph_classification':  # ogbg-molhiv, ogbg-molpcba
                    output = output.squeeze(-1)
                    target_y = y.squeeze(-1).float()
                    criterion = torch.nn.BCEWithLogitsLoss()
                else:
                    target_y = y.view_as(output)

            if task_type == 'node_classification':
                loss = criterion(output, target_y)
                p_val = F.log_softmax(output, dim=1)
                if rocauc_flag:  # gen-cls
                    rocauc = eval_rocauc(target_y, p_val)
                    total_rocauc += rocauc
                else:
                    # acc = eval_acc(target_y, p_val)
                    acc = accuracy(output, target_y).to(device)
                    total_acc += acc.item()
            elif task_type == 'graph_classification':  # ogbg-molhiv, ogbg-molpcba
                if not rocauc_flag:  # i.e., ogbg-molpcba
                    # NOTE: The draft model outputs a tensor of shape [batch_size, 1],
                    # while the `target_y` dimension is 128. To align dimensions for loss calculation,
                    # we initially expanded the output (i.e., the commented code below):
                    # `output_tmp = output.unsqueeze(-1).expand(-1, 128)`
                    # This was later fixed in the current implementation.
                    is_labeled = target_y == target_y
                    # output_ls = output_tmp[is_labeled]
                    output_ls = output[is_labeled]
                    target_y_ls = target_y[is_labeled]
                    loss = criterion(output_ls, target_y_ls)
                else:
                    loss = criterion(output, target_y)

                p_val = torch.sigmoid(output)
                if p_val.dim() == 1 and target_y.dim() == 1:
                    p_val = p_val.unsqueeze(1)
                    target_y = target_y.unsqueeze(1)

                if not rocauc_flag:
                    # p_val = p_val.unsqueeze(-1).expand(-1, 128)
                    ap_batch = evaluator.eval({'y_pred': p_val, 'y_true': target_y})['ap']
                    total_ap += ap_batch
                else:
                    all_clf_logits.append(p_val)
                    all_clf_labels.append(target_y)
            else:
                loss = model.mse_loss(output, target_y)
                total_mae += model.mae(output, target_y).item()

            total_loss += loss.item() * data.num_graphs

        if task_type == 'node_classification':
            if rocauc_flag:
                return total_loss / total_len, total_rocauc / batch_cnt
            else:
                return total_loss / total_len, total_acc / batch_cnt
        elif task_type == 'graph_classification':
            if rocauc_flag:
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)
                rocauc = evaluator.eval({'y_pred': all_clf_logits, 'y_true': all_clf_labels})['rocauc']
                return total_loss / total_len, rocauc
            else:  # i.e., ogbg-molpcba
                return total_loss / total_len, total_ap / batch_cnt
        else:
            return total_loss / total_len, total_mae / total_len


# For graph regression / node regression / graph classification / node classification
# sampling_type = 1, 2, 3, 4
def test_with_st_1234(model, test_loader, k_list, transform, target_index,
                      total_len, device, task_type, rocauc_flag=False):
    model.eval()
    with torch.no_grad():
        total_mae = 0
        total_loss = 0
        total_acc = 0
        total_rocauc = 0
        total_ap = 0
        all_clf_labels = []
        all_clf_logits = []
        batch_cnt = len(test_loader)
        for data in tqdm(test_loader, desc='Processing test_loader batches'):
            data = transform(data)
            x = data.x.to(device)
            if x.dtype == torch.int64:
                x = x.float()
            edge_index = data.edge_index.to(device)
            edge_weight = data.edge_weight.to(device)
            edge_data = (edge_index, edge_weight)
            y = data.y.to(device)
            batch = data.batch.to(device)
            output = model(x, k_list, edge_data, batch)
            if target_index != -1:
                target_y = y[:, target_index].view_as(output)
            else:
                if task_type == 'node_classification':
                    target_y = y.squeeze(-1)
                    criterion = torch.nn.CrossEntropyLoss()
                elif task_type == 'graph_classification':  # ogbg-molhiv, ogbg-molpcba
                    output = output.squeeze(-1)
                    target_y = y.squeeze(-1).float()
                    criterion = torch.nn.BCEWithLogitsLoss()
                else:
                    target_y = y.view_as(output)

            if task_type == 'node_classification':
                loss = criterion(output, target_y)
                p_test = F.log_softmax(output, dim=1)
                if rocauc_flag:  # gen-cls
                    rocauc = eval_rocauc(target_y, p_test)
                    total_rocauc += rocauc
                else:
                    # acc = eval_acc(target_y, p_test)
                    acc = accuracy(output, target_y).to(device)
                    total_acc += acc.item()
            elif task_type == 'graph_classification':  # ogbg-molhiv, ogbg-molpcba
                if not rocauc_flag:  # i.e., ogbg-molpcba
                    # NOTE: The draft model outputs a tensor of shape [batch_size, 1],
                    # while the `target_y` dimension is 128. To align dimensions for loss calculation,
                    # we initially expanded the output (i.e., the commented code below):
                    # `output_tmp = output.unsqueeze(-1).expand(-1, 128)`
                    # This was later fixed in the current implementation.
                    is_labeled = target_y == target_y
                    # output_ls = output_tmp[is_labeled]
                    output_ls = output[is_labeled]
                    target_y_ls = target_y[is_labeled]
                    loss = criterion(output_ls, target_y_ls)
                else:
                    loss = criterion(output, target_y)

                p_test = torch.sigmoid(output)
                if p_test.dim() == 1 and target_y.dim() == 1:
                    p_test = p_test.unsqueeze(1)
                    target_y = target_y.unsqueeze(1)

                if not rocauc_flag:
                    # p_test = p_test.unsqueeze(-1).expand(-1, 128)
                    ap_batch = evaluator.eval({'y_pred': p_test, 'y_true': target_y})['ap']
                    total_ap += ap_batch
                else:
                    all_clf_logits.append(p_test)
                    all_clf_labels.append(target_y)
            else:
                loss = model.mse_loss(output, target_y)
                total_mae += model.mae(output, target_y).item()

            total_loss += loss.item() * data.num_graphs

        if task_type == 'node_classification':
            if rocauc_flag:
                return total_loss / total_len, total_rocauc / batch_cnt
            else:
                return total_loss / total_len, total_acc / batch_cnt
        elif task_type == 'graph_classification':
            if rocauc_flag:
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)
                rocauc = evaluator.eval({'y_pred': all_clf_logits, 'y_true': all_clf_labels})['rocauc']
                return total_loss / total_len, rocauc
            else:  # ogbg-molpcba
                return total_loss / total_len, total_ap / batch_cnt
        else:
            return total_loss / total_len, total_mae / total_len


def setup_logger(log_file_name):
    log_dir = "./logs"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_dir, log_file_name)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
