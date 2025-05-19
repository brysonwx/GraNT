import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


def grant_sampler(model, data_loader, batch_size, k_list, transform, target_index,
                  sampling_ratio, sampling_type, task_type, rocauc_flag, device):
    assert sampling_type in (1, 2, 3, 4)
    with torch.no_grad():
        total_len = len(data_loader.dataset)
        if sampling_type in (2, 4):
            sampling_nums = math.ceil(total_len * sampling_ratio)
            sampling_per_batch = _get_sampling_per_batch(total_len, batch_size, sampling_nums)
        subset_topk_idxs = []  # the final sampled global idxs from the data_loader.dataset
        diffs = []
        cnt = 0
        for data in tqdm(data_loader, desc='Processing grant_sampler data_loader batches'):
            data = transform(data)
            x = data.x.to(device)
            if x.dtype == torch.int64:
                x = x.float()
            edge_index = data.edge_index.to(device)
            edge_weight = data.edge_weight.to(device)
            edge_data = (edge_index, edge_weight)
            y = data.y.to(device)

            batch = data.batch.to(device)
            num_graphs = batch.max().item() + 1
            rows_filtered = None  # i.e., for ogbg-molpcba

            output = model(x, k_list, edge_data, batch)
            if target_index != -1:
                target_y = y[:, target_index].view_as(output)
            else:
                if task_type == 'node_classification':
                    target_y = y.squeeze(-1)
                elif task_type == 'graph_classification':
                    output = output.squeeze(-1)
                    target_y = y.squeeze(-1).float()
                    if not rocauc_flag:  # i.e., ogbg-molpcba
                        rows = torch.arange(target_y.shape[0]).unsqueeze(1).expand_as(target_y).to(device)
                        is_labeled = target_y == target_y
                        rows_filtered = rows[is_labeled]  # i.e., for ogbg-molpcba
                        output = output[is_labeled]
                        target_y = target_y[is_labeled]
                        # target_y = torch.nanmean(target_y, dim=1, keepdim=True).squeeze(-1)
                else:
                    target_y = y.view_as(output)

            if sampling_type == 1:  # graph regression / classification
                # 1. sample some [batches] of graphs according to batch abs_diff
                if task_type == 'graph_classification':
                    output = F.softmax(output, dim=0)
                sub_diffs = torch.abs(target_y - output).squeeze()
                diffs.append(torch.mean(sub_diffs))
            elif sampling_type == 2:  # graph regression / classification
                # 2. sample some [graphs] in each batch according to single graph abs_diff
                one_batch_diffs = []
                if task_type == 'graph_classification':
                    output = F.softmax(output, dim=0)

                if rows_filtered is not None:  # i.e., for ogbg-molpcba
                    for i in range(num_graphs):
                        mask = (rows_filtered == i)
                        one_graph_num_tasks = mask.sum().item()
                        one_graph_output = output[mask]
                        one_graph_target_y = target_y[mask]
                        one_graph_diff = torch.norm(one_graph_target_y - one_graph_output, p=2)
                        # one_graph_diff = torch.sum(torch.abs(one_graph_target_y - one_graph_output))
                        one_batch_diffs.append(one_graph_diff / one_graph_num_tasks)
                    one_batch_diffs = torch.tensor(one_batch_diffs).to(device)
                else:
                    one_batch_diffs = torch.abs(target_y - output).squeeze()
                global_sub_topk_idxs = _get_global_sub_topk_idxs(
                    one_batch_diffs, sampling_per_batch[cnt], cnt, batch_size)
                subset_topk_idxs.extend(global_sub_topk_idxs)
            elif sampling_type == 3:  # node regression / classification
                # 3. sample some [batches] of graphs according to mean batch l2_norm diff
                need_one_hot = False
                if task_type == 'node_classification':
                    output = F.softmax(output, dim=1)
                    need_one_hot = True
                one_batch_diffs = _get_one_batch_diffs(num_graphs, batch, output, target_y, need_one_hot)
                one_batch_mean_diff = sum(one_batch_diffs) / num_graphs
                diffs.append(one_batch_mean_diff)
            elif sampling_type == 4:  # node regression / classification
                # 4. sample some [graphs] in each batch according to single graph l2_norm diff
                need_one_hot = False
                if task_type == 'node_classification':
                    output = F.softmax(output, dim=1)
                    need_one_hot = True
                num_graphs = batch.max().item() + 1
                one_batch_diffs = _get_one_batch_diffs(num_graphs, batch, output, target_y, need_one_hot)
                global_sub_topk_idxs = _get_global_sub_topk_idxs(
                    torch.tensor(one_batch_diffs), sampling_per_batch[cnt], cnt, batch_size)
                subset_topk_idxs.extend(global_sub_topk_idxs)

            cnt += 1

        if sampling_type in (1, 3):
            topk_nums = math.ceil(sampling_ratio * len(diffs))
            processed_idxs = []
            stacked_diffs = torch.stack(diffs)
            topk_diffs, topk_idxs = torch.topk(stacked_diffs, topk_nums)
            sorted_topk_idxs, _ = torch.sort(topk_idxs)

            for idx in sorted_topk_idxs:
                idx_item = idx.item()
                if idx_item != cnt - 1:  # not the last batch
                    processed_idxs.append(torch.arange(idx * batch_size, (idx + 1) * batch_size))
                else:  # the last batch
                    processed_idxs.append(torch.arange(idx * batch_size, total_len))

            subset_topk_idxs = torch.cat(processed_idxs)

        subset_data = Subset(data_loader.dataset, subset_topk_idxs)
        subset_loader = DataLoader(subset_data, batch_size=batch_size, shuffle=False)

        return subset_loader


def _get_sampling_per_batch(total_len, batch_size, sampling_nums):
    num_batches = (total_len + batch_size - 1) // batch_size
    batch_sizes = [batch_size] * (num_batches - 1) + [total_len % batch_size or batch_size]
    sampling_per_batch = []
    cumulative_error = 0
    for size in batch_sizes:
        exact_value = size / total_len * sampling_nums
        adjusted_value = int(exact_value + cumulative_error)
        sampling_per_batch.append(adjusted_value)
        cumulative_error += (exact_value - adjusted_value)

    return sampling_per_batch


def _get_one_batch_diffs(num_graphs, batch, output, target_y, need_one_hot):
    one_batch_diffs = []
    for i in range(num_graphs):
        mask = (batch == i)
        one_graph_num_nodes = mask.sum().item()
        one_graph_output = output[mask]
        one_graph_target_y = target_y[mask]
        if need_one_hot:
            one_graph_target_y = F.one_hot(one_graph_target_y, num_classes=10)
        one_graph_diff = torch.norm(one_graph_target_y - one_graph_output, p=2)
        # one_graph_diff = torch.sum(torch.abs(one_graph_target_y - one_graph_output))
        one_batch_diffs.append(one_graph_diff / one_graph_num_nodes)

    return one_batch_diffs


def _get_global_sub_topk_idxs(one_batch_diffs, sampling_cnt, cnt, batch_size):
    topk_sub_diffs, topk_sub_idxs = torch.topk(one_batch_diffs, sampling_cnt)
    start_global_idx = cnt * batch_size
    sorted_sub_topk_idxs, _ = torch.sort(topk_sub_idxs)
    global_sub_topk_idxs = [start_global_idx + idx for idx in sorted_sub_topk_idxs]

    return global_sub_topk_idxs
