import yaml
import time
import wandb
import torch
import argparse
import faulthandler
import os.path as osp
from copy import deepcopy
from yaml import SafeLoader
import torch.nn as nn
import torch.optim as optim
from models import GrantGCN
from scheduler import scheduler_factory
from strategy import strategy_factory
from sampler import grant_sampler
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import GCNNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import setup_logger, seed_everything, get_cls_dataset, get_reg_dataset, modify_mlp_dims
from utils import train_with_st_1234, validate_with_st_1234, test_with_st_1234

torch.autograd.set_detect_anomaly(True)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# import setproctitle
# setproctitle.setproctitle('python')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_layers', type=int, default=2)
    # k belongs to [0, 1, 2,...], i.e., [I, A^(1), A^(2)]--> k = 3
    parser.add_argument('--k_list', type=list, default=[3, 2])
    parser.add_argument('--dataset', type=str, default='QM9')
    # NOTE: The first dim should be (k_list[0] * node dim)
    parser.add_argument('--mlp_dims', type=list, default=[[33, 19], ])
    parser.add_argument('--out_dim', type=int, default=19)
    # use SparseTensor, True
    parser.add_argument('--sparse', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    # node_regression, graph_regression, node_classification, graph_classification
    parser.add_argument('--task_type', type=str, default='graph_regression')
    # for some regression tasks,
    # e.g., choose one of 19 different properties of QM9 for regression, default: 0
    parser.add_argument('--target_index', type=int, default=0)
    parser.add_argument('--clip_max', type=float, default=1000)
    parser.add_argument('--sampling_flag', type=bool, default=True)  # whether to sample
    parser.add_argument('--scheduler_type', type=str, default='step')
    parser.add_argument('--start_ratio', type=float, default=0.05)
    parser.add_argument('--strategy_type', type=str, default='simple_curriculum')
    parser.add_argument('--sampling_type', type=int, default=1)
    parser.add_argument('--config_file', type=str, default='/home/ubuntu/codes/GraNT/config.yaml')
    parser.add_argument('--log_file', type=str, default='default.log', help='log file name')
    args = parser.parse_args()
    return args


def main_grant():
    args = get_args()
    dataset_name = args.dataset
    config = yaml.load(open(args.config_file), Loader=SafeLoader)[dataset_name]

    logger = setup_logger(config['log_file'])
    logger.info(f'config data: {config} for dataset: {dataset_name}')

    seed_everything(config['seed'])

    device = torch.device(f"cuda:{config['device_id']}") if torch.cuda.is_available() else torch.device('cpu')

    task_type = config['task_type']
    assert task_type in ('node_regression', 'graph_regression', 'node_classification', 'graph_classification')

    sampling_flag = config['sampling_flag']
    sampling_type = config['sampling_type']
    # see more details of the sampling_type: 1, 2, 3, 4, 5 in `sampler.py`
    # 1, 2: sample some graphs, graph regression / classification tasks from batches of graphs
    # 3, 4: sample some graphs, node regression /classification tasks from batches of graphs
    sampling_type_flag_1234 = sampling_type in (1, 2, 3, 4)
    assert sampling_type_flag_1234

    # transform graph data into the one that holds normalized adjacency matrix
    # see: https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.transforms.GCNNorm.html
    transform = GCNNorm(add_self_loops=False)

    atom_enc_flag = False
    ogbg_mol_flag = dataset_name in ('ogbg-molhiv', 'ogbg-molpcba')
    rocauc_flag = False
    if (task_type == 'graph_classification' and dataset_name == 'ogbg-molhiv') or \
            (task_type == 'node_classification' and dataset_name == 'gen-cls'):  # binary classification
        rocauc_flag = True

    dataset_path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)

    if task_type in ('node_regression', 'graph_regression'):
        dataset = get_reg_dataset(dataset_path, dataset_name)
        x_dim = dataset[0].x.size(1)  # the node dimension
        if task_type == 'node_regression':
            if dataset_name == 'gen-reg':
                assert sampling_type in (3, 4)
                train_dataset = dataset[:30000]
                val_dataset = dataset[30000:40000]
                test_dataset = dataset[40000:]
            else:
                raise NotImplementedError
        else:  # graph_regression
            assert sampling_type in (1, 2)
            if dataset_name == 'QM9':
                train_dataset = dataset[:110000]
                val_dataset = dataset[110000:120000]
                test_dataset = dataset[120000:]
            elif dataset_name == 'ZINC':
                base_path = osp.expanduser('~/datasets')
                train_dataset = dataset  # only train
                val_dataset = ZINC(root=osp.join(base_path, 'ZINC-val'), split='val')
                test_dataset = ZINC(root=osp.join(base_path, 'ZINC-test'), split='test')
            else:
                raise NotImplementedError

        batch_size = config['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train_total_len = len(train_loader.dataset)
        val_total_len = len(val_loader.dataset)
        test_total_len = len(test_loader.dataset)
    else:  # node_classification, graph_classification
        dataset = get_cls_dataset(dataset_path, dataset_name)
        if task_type == 'node_classification':
            if sampling_type_flag_1234:  # diverse graphs
                if dataset_name == 'gen-cls':
                    assert sampling_type in (3, 4)
                    data = dataset[0]
                    x_dim = data.x.size(1)
                    train_dataset = dataset[:30000]
                    val_dataset = dataset[30000:40000]
                    test_dataset = dataset[40000:]
                else:
                    raise NotImplementedError

                batch_size = config['batch_size']
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        else:  # graph_classification, diverse graphs
            if dataset_name in ('ogbg-molhiv', 'ogbg-molpcba'):
                assert sampling_type in (1, 2)
                x_dim = dataset[0].x.size(1)
                batch_size = config['batch_size']
                split_idx = dataset.get_idx_split()
                train_loader = DataLoader(dataset[split_idx['train']], batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(dataset[split_idx['valid']], batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(dataset[split_idx['test']], batch_size=batch_size, shuffle=False)
                atom_enc_flag = True

        train_total_len = len(train_loader.dataset)
        val_total_len = len(val_loader.dataset)
        test_total_len = len(test_loader.dataset)

    sparse = config['sparse']
    n_layers = config['n_layers']
    k_list = config['k_list']
    assert n_layers == len(k_list)

    mlp_dims = config['mlp_dims']
    in_dim = modify_mlp_dims(sampling_type, mlp_dims, n_layers, k_list, x_dim)
    out_dim = config['out_dim']
    last_linear_dims = (in_dim, out_dim)

    wandb_config = deepcopy(config)
    wandb_config['dataset'] = dataset_name
    wandb.init(
        project='grant',
        config=wandb_config
    )

    grant_gcn = GrantGCN(
        n_layers, x_dim, mlp_dims, last_linear_dims,
        task_type, sparse, device, atom_enc_flag, dataset_name)
    grant_gcn = grant_gcn.to(device)
    optimizer = optim.AdamW(grant_gcn.parameters(), lr=config['lr'], weight_decay=float(config['weight_decay']))
    for param in grant_gcn.parameters():
        if param.dim() > 1:
            # nn.init.xavier_uniform_(param)
            nn.init.kaiming_uniform_(param, nonlinearity='relu')

    if ogbg_mol_flag:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=7, verbose=True, min_lr=1e-5)

    if sampling_flag:  # invoke the sampling
        ratio_scheduler = scheduler_factory(config['scheduler_type'])
        strategy = strategy_factory(config['strategy_type'])


    # start training
    start_time = time.time()
    # NOTE: the first train / validation with the initial model
    mae_train = None
    acc_train = None
    if sampling_type_flag_1234:
        if task_type in ('node_classification', 'graph_classification'):
            loss_train, acc_train = train_with_st_1234(
                grant_gcn, optimizer, train_loader, k_list, transform, config['target_index'],
                train_total_len, config['clip_max'], task_type, device, rocauc_flag=rocauc_flag)
        else:
            loss_train, mae_train = train_with_st_1234(grant_gcn, optimizer, train_loader, k_list, transform, config['target_index'],
                                                       train_total_len, config['clip_max'], task_type, device, rocauc_flag=rocauc_flag)

    wandb_data = {
        'loss_train': loss_train,
        'wallclock_time': 0,
    }
    if mae_train:
        wandb_data['mae_train'] = mae_train
    if acc_train:
        wandb_data['acc_train'] = acc_train
    wandb.log(wandb_data)

    mae_val = None
    acc_val = None
    if sampling_type_flag_1234:
        if task_type in ('node_classification', 'graph_classification'):
            loss_val, acc_val = validate_with_st_1234(
                grant_gcn, val_loader, k_list, transform, config['target_index'],
                val_total_len, device, task_type, rocauc_flag=rocauc_flag)
        else:
            loss_val, mae_val = validate_with_st_1234(
                grant_gcn, val_loader, k_list, transform, config['target_index'],
                val_total_len, device, task_type, rocauc_flag=rocauc_flag)

    wandb_data = {
        'loss_val': loss_val,
        'wallclock_time': 0,
    }
    if mae_val:
        wandb_data['mae_val'] = mae_val
    if acc_val:
        wandb_data['acc_val'] = acc_val
    wandb.log(wandb_data)

    for epoch in range(config['epochs']):
        start_all_train_time = time.time()
        wallclock_time = (time.time() - start_time)
        if sampling_flag:
            start_sampling_time = time.time()
            sampling_ratio = ratio_scheduler(epoch, config['epochs'], config['start_ratio'])
            mt_flag, mt_intervals = strategy(epoch, config['epochs'], startup_ratio=config['start_ratio'])
            # i.e., the full training, no sampling
            if sampling_ratio and float(sampling_ratio) == 1.0:
                if sampling_type_flag_1234:
                    data_loader = train_loader
                logger.info(
                    f'Epoch: {epoch + 1:04d}, mt_flag: {mt_flag}, mt_intervals: {mt_intervals}, '
                    f'sampling_ratio: {sampling_ratio}, train_total_len: {train_total_len}, [xxx_flag]'
                )
            else:
                if mt_flag:  # i.e., resampling with a new sampling_ratio
                    data_loader = grant_sampler(
                        grant_gcn, train_loader, batch_size, k_list, transform,
                        config['target_index'], sampling_ratio, config['sampling_type'], task_type, rocauc_flag, device)
                elif not mt_flag and mt_intervals is None:  # i.e., no mt_flag at all, just use the train_loader.
                    data_loader = train_loader
                else:  # 1) not mt_flag & 2) mt_intervals is not None. use last_data_loader.
                    data_loader = last_data_loader
                last_data_loader = data_loader

                if mt_flag:
                    train_total_len = len(data_loader.dataset)
                    end_sampling_time = time.time()
                    sampling_time = end_sampling_time - start_sampling_time
                    logger.info(
                        f'Epoch: {epoch + 1:04d}, mt_flag: {mt_flag}, mt_intervals: {mt_intervals}, '
                        f'sampling_ratio: {sampling_ratio}, train_total_len: {train_total_len}, sampling_time: {sampling_time}'
                    )
                else:
                    logger.info(
                        f'Epoch: {epoch + 1:04d}, mt_flag: {mt_flag}, mt_intervals: {mt_intervals}, '
                        f'sampling_ratio: {sampling_ratio}, train_total_len: {train_total_len}'
                    )

        else:  # full training, no sampling
            data_loader = train_loader
            train_total_len = len(data_loader.dataset)

        start_only_train_time = time.time()
        mae_train = None
        acc_train = None
        if sampling_type_flag_1234:
            if task_type in ('node_classification', 'graph_classification'):
                loss_train, acc_train = train_with_st_1234(grant_gcn, optimizer, data_loader, k_list,
                                                           transform, config['target_index'], train_total_len,
                                                           config['clip_max'], task_type, device, rocauc_flag=rocauc_flag)
            else:
                loss_train, mae_train = train_with_st_1234(grant_gcn, optimizer, data_loader, k_list,
                                                           transform, config['target_index'], train_total_len,
                                                           config['clip_max'], task_type, device, rocauc_flag=rocauc_flag)

        if task_type in ('node_regression', 'graph_regression'):
            train_metric_item = {
                'mae_train': mae_train,
            }
        elif task_type in ('node_classification', 'graph_classification'):
            train_metric_item = {
                'acc_train': acc_train,
            }

        end_only_train_time = time.time()

        logger.info(
            f'[train]task_type: [{task_type}], sampling_type: [{sampling_type}], '
            f'epoch: {epoch + 1:04d}, loss_item: {loss_train:.6f}, metric_item: {train_metric_item}, '
            f'only_train_time: {end_only_train_time - start_only_train_time:.4f}s '
        )
        end_all_train_time = time.time()

        if epoch % 1 == 0:
            mae_val = None
            acc_val = None
            start_validate_time = time.time()
            if sampling_type_flag_1234:
                if task_type in ('node_classification', 'graph_classification'):
                    loss_val, acc_val = validate_with_st_1234(
                        grant_gcn, val_loader, k_list, transform, config['target_index'],
                        val_total_len, device, task_type, rocauc_flag=rocauc_flag)
                    if ogbg_mol_flag:
                        scheduler_lr_threshold = config['scheduler_lr_threshold']
                        scheduler.step(acc_val)
                        current_lr = optimizer.param_groups[0]['lr']
                        if sampling_flag and mt_flag:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = max(scheduler_lr_threshold, current_lr)
                        logger.info(f"Epoch {epoch + 1}: acc_train={acc_train:.4f}, acc_val={acc_val:.4f}, lr={current_lr:.6f}")
                else:
                    loss_val, mae_val = validate_with_st_1234(
                        grant_gcn, val_loader, k_list, transform, config['target_index'],
                        val_total_len, device, task_type, rocauc_flag=rocauc_flag)

            if task_type in ('node_regression', 'graph_regression'):
                val_metric_item = {
                    'mae_val': mae_val,
                }
            elif task_type in ('node_classification', 'graph_classification'):
                val_metric_item = {
                    'acc_val': acc_val,
                }

            end_validate_time = time.time()
            logger.info(
                f'[validate]task_type: [{task_type}], sampling_type: [{sampling_type}], '
                f'epoch: {epoch + 1:04d}, loss_item: {loss_val:.6f}, '
                f'metric_item: {val_metric_item}, time: {end_validate_time - start_validate_time:.4f}s'
            )

        wandb_data = {
            'epoch': epoch,
            'loss_train': loss_train,
            'all_train_time': end_all_train_time - start_all_train_time,
            'only_train_time': end_only_train_time - start_only_train_time,
            'loss_val': loss_val,
            'validate_time': end_validate_time - start_validate_time,
            'wallclock_time': wallclock_time,
        }
        if mae_train:
            wandb_data['mae_train'] = mae_train
        if acc_train:
            wandb_data['acc_train'] = acc_train
        if mae_val:
            wandb_data['mae_val'] = mae_val
        if acc_val:
            wandb_data['acc_val'] = acc_val
        wandb.log(wandb_data)

    start_test_time = time.time()
    if sampling_type_flag_1234:
        if task_type in ('node_classification', 'graph_classification'):
            loss_test, acc_test = test_with_st_1234(
                grant_gcn, test_loader, k_list, transform, config['target_index'],
                test_total_len, device, task_type, rocauc_flag=rocauc_flag)
        else:
            loss_test, mae_test = test_with_st_1234(
                grant_gcn, test_loader, k_list, transform, config['target_index'],
                test_total_len, device, task_type, rocauc_flag=rocauc_flag)

    if task_type in ('node_regression', 'graph_regression'):
        test_metric_item = {
            'mae_test': mae_test,
        }
    elif task_type in ('node_classification', 'graph_classification'):
        test_metric_item = {
            'acc_test': acc_test,
        }

    end_test_time = time.time()
    logger.info(
        f'[test]task_type: [{task_type}], sampling_type: [{sampling_type}], '
        f'loss_item: {loss_test:.6f}, metric_item: {test_metric_item}, '
        f'time: {end_test_time - start_test_time:.4f}s'
    )


if __name__ == '__main__':
    faulthandler.enable()
    main_grant()
