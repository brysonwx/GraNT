# Steps to reproduce ogbg-molhiv leaderboard submission using GIN:
# (for https://web.archive.org/web/20240822032633/https://ogb.stanford.edu/docs/leader_graphprop/#ogbg-molhiv )

# 1) Install dependencies (run `install_dependencies.sh` this comes with or commands below):
#```
#pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
#pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.2.1+cu121.html
#pip install torch-geometric # I'm using 2.5.3 right now
#pip install ogb # I'm using 1.3.6 right now
#```

# 2) Run this script `python main_gin.py` (I'm using python 3.10.12 but should be flexible)

# Author: William Bruns (adde.animulis@gmail.com)
# Credit to Stanford XCS224W, they had a homework assignment using this dataset which this solution was inspired by (the OGB leaderboard was not mentioned in and is not part of the course).
# Note, this uses the atom and not the edge features of the dataset. This is a work in progress and at time of writing I have been working on this for ~2 days, so haven't tried that yet. This same code will work if the dataset id is changed for ogbg-molpcba -- I am adding that next and attempting to pretrain on that to improve performance on this right now.

import os
import argparse
import pickle
import torch
import logging
import torch_geometric
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import MLP
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
import pandas as pd
import copy
import numpy as np
import random
from tqdm import tqdm
from gin_sampler import grant_sampler
from gin_scheduler import scheduler_factory
from gin_strategy import strategy_factory
from torch_geometric.transforms import GCNNorm

import setproctitle
setproctitle.setproctitle('python')

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_id", type=str, default='ogbg-molhiv')
argparser.add_argument("--device", type=str, default='cuda:0')
argparser.add_argument("--num_layers", type=int, default=2)
argparser.add_argument("--hidden_dim", type=int, default=64)
argparser.add_argument("--learning_rate", type=float, default=0.001)
argparser.add_argument("--dropout_p", type=float, default=0.5)
argparser.add_argument("--epochs", type=int, default=600)
argparser.add_argument("--batch_size", type=int, default=500)
argparser.add_argument("--weight_decay", type=float, default=1e-6)
argparser.add_argument("--random_seed", type=int, default=114514)
argparser.add_argument("--hide_test_metric", action="store_true", default=False)
argparser.add_argument("--sampling_flag", type=bool, default=False)
argparser.add_argument("--sampling_type", type=int, default=1)
argparser.add_argument('--log_file', type=str, default='ogbg-molhiv.log', help='log file name')
args = argparser.parse_args()

# Let's set a random seed for reproducibility
# -------------------------------------------
# If using a GPU choosing the same seed cannot be used to guarantee
# that one gets the same result from run to run,
# but may still be useful to ensure one is starting with different seeds.
# The author used a CPU and random seeds 0..9 inclusive for their
# leaderboard submission (nothing impressive, just to practice and participate!
# decision tree on molecular fingerprints is more efficient for this task!)
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file_name='log.txt'):
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

# Set this to 'cpu' if you NEED to reproduce exact numbers.
device = args.device #'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
set_seeds(args.random_seed)

# defaults in comments below get
# rocauc 0.7835 +/- 0.0125 (mean +/- sample std, n=10;
# random seeds 0..9 inclusive) on ogbg-molhiv
# (n.b. stopping at 40 epochs would have improved the score,
# but I had chosen 50 epochs and now have observed the result)
config = {
 'device': args.device,
 # must be valid ogb dataset id, e.g. ogbg-molhiv, ogbg-molpcba, etc
 'dataset_id': args.dataset_id,
 'num_layers': args.num_layers,#2,
 'hidden_dim': args.hidden_dim,#64,
 'dropout': args.dropout_p,#0.5,
 'learning_rate': args.learning_rate,#0.001,
 'epochs': args.epochs,#50,
 'batch_size': args.batch_size,#32,
 'weight_decay': args.weight_decay, #1e-6
 'start_ratio': 0.1,
 'scheduler_type': 'step',
 'strategy_type': 'simple_curriculum',
 'target_index': 0,
 'sampling_type': args.sampling_type,
 'log_file': args.log_file,
 'random_seed': args.random_seed,
}

logger = setup_logger(config['log_file'])
logger.info(f"config: {config}")

sampling_flag = args.sampling_flag  # False, True
logger.info(f'sampling_flag: {sampling_flag}')

transform = GCNNorm(add_self_loops=False)

if args.dataset_id != 'ogbg-molhiv' and args.dataset_id != 'ogbg-molpcba':
    raise NotImplementedError("Unsupported ogb dataset id! Please try either 'ogbg-molhiv' (single task, ROCAUC metric) or 'ogbg-molpcba' (multi-task, AP metric)")

# dataset loading
dataset = PygGraphPropPredDataset(name=config["dataset_id"], transform=None)
evaluator = Evaluator(name=config["dataset_id"])
split_idx = dataset.get_idx_split()

train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config["batch_size"], shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=config["batch_size"], shuffle=False)
test_loader = None
if not args.hide_test_metric:
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=config["batch_size"], shuffle=False)
# end dataset loading

# computes a node embedding using GINConv layers, then uses pooling to predict graph level properties
class GINGraphPropertyModel(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout_p):
      super(GINGraphPropertyModel, self).__init__()
      # fields used for computing node embedding
      self.node_encoder = AtomEncoder(hidden_dim)
        
      self.convs = torch.nn.ModuleList(
          [torch_geometric.nn.conv.GINConv(MLP([hidden_dim, hidden_dim, hidden_dim])) for idx in range(0, num_layers)]
      )
      self.bns = torch.nn.ModuleList(
          [torch.nn.BatchNorm1d(num_features = hidden_dim) for idx in range(0, num_layers - 1)]
      )
      self.dropout_p = dropout_p
      # end fields used for computing node embedding
      # fields for graph embedding
      self.pool = global_add_pool
      self.linear_hidden = torch.nn.Linear(hidden_dim, hidden_dim)
      self.linear_out = torch.nn.Linear(hidden_dim, output_dim)
      # end fields for graph embedding
    def reset_parameters(self):
      for conv in self.convs:
        conv.reset_parameters()
      for bn in self.bns:
        bn.reset_parameters()
      self.linear_hidden.reset_parameters()
      self.linear_out.reset_parameters()

    def forward(self, batched_data):
      x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
      # compute node embedding
      x = self.node_encoder(x)
      for idx in range(0, len(self.convs)):
        x = self.convs[idx](x, edge_index)
        if idx < len(self.convs) - 1:
          x = self.bns[idx](x)
          x = torch.nn.functional.relu(x)
          x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      # note x is raw logits, NOT softmax'd
      # end computation of node embedding
      # convert node embedding to a graph level embedding using pooling
      x = self.pool(x, batch)
      x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      # transform the graph embedding to the output dimension
      # MLP after graph embed ensures we are not requiring the raw pooled node embeddings to be linearly separable
      x = self.linear_hidden(x)
      x = torch.nn.functional.relu(x)
      x = torch.nn.functional.dropout(x, self.dropout_p, training=self.training)
      out = self.linear_out(x)
      return out

# can be used with multiple task outputs (like for molpcba) or single task output; 
# and supports using just the first output of a multi-task model if applied to a single task (for pretraining molpcba and transferring to molhiv)
def train(model, device, data_loader, optimizer, loss_fn):
  model.train()
  for step, batch in enumerate(tqdm(data_loader, desc="Training batch")):
    batch = batch.to(device)
    if batch.x.shape[0] != 1 and batch.batch[-1] != 0:
      # ignore nan targets (unlabeled) when computing training loss.
      non_nan = batch.y == batch.y
      loss = None
      optimizer.zero_grad()
      out = model(batch)
      non_nan = non_nan[:min(non_nan.shape[0], out.shape[0])]
      batch_y = batch.y[:out.shape[0], :]
      # for crudely adapting multitask models to single task data
      if batch.y.shape[1] == 1:
        out = out[:, 0]
        batch_y = batch_y[:, 0]
        non_nan = batch_y == batch_y
        loss = loss_fn(out[non_nan].reshape(-1, 1)*1., batch_y[non_nan].reshape(-1, 1)*1.)
      else:
        loss = loss_fn(out[non_nan], batch_y[non_nan])
      loss.backward()
      optimizer.step()
  return loss.item()

def eval(model, device, loader, evaluator, save_model_results=False, save_filename=None):
  model.eval()
  y_true = []
  y_pred = []
  for step, batch in enumerate(tqdm(loader, desc="Evaluation batch")):
      batch = batch.to(device)
      if batch.x.shape[0] == 1:
          pass
      else:
          with torch.no_grad():
              pred = model(batch)
              # for crudely adapting multitask models to single task data
              if batch.y.shape[1] == 1:
                pred = pred[:, 0]
              batch_y = batch.y[:min(pred.shape[0], batch.y.shape[0])]
              y_true.append(batch_y.view(pred.shape).detach().cpu())
              y_pred.append(pred.detach().cpu())
  y_true = torch.cat(y_true, dim=0).numpy()
  y_pred = torch.cat(y_pred, dim=0).numpy()
  input_dict = {"y_true": y_true.reshape(-1, 1) if batch.y.shape[1] == 1 else y_true, "y_pred": y_pred.reshape(-1, 1) if batch.y.shape[1] == 1 else y_pred}
  if save_model_results:
      single_task = len(y_true.shape) == 1 or y_true.shape[1] == 1
      if single_task:
          data = {
              'y_pred': y_pred.squeeze(),
              'y_true': y_true.squeeze()
          }
          pd.DataFrame(data=data).to_csv('ogbg_graph_' + save_filename + '.csv', sep=',', index=False)
      else:
          num_tasks = y_true.shape[1]
          for task_idx in range(num_tasks):
              data = {
                  'y_pred': y_pred[:, task_idx].squeeze(),
                  'y_true': y_true[:, task_idx].squeeze()
              }
              pd.DataFrame(data=data).to_csv('ogbg_graph_' + save_filename + f'_task_{task_idx}.csv', sep=',', index=False)
  return evaluator.eval(input_dict)

model = GINGraphPropertyModel(config['hidden_dim'], dataset.num_tasks, config['num_layers'], config['dropout']).to(device)
print(f"parameter count: {sum(p.numel() for p in model.parameters())}")
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
loss_fn = torch.nn.BCEWithLogitsLoss()
best_model = None
best_valid_metric_at_save_checkpoint = 0
best_train_metric_at_save_checkpoint = 0

if sampling_flag:  # invoke the sampling
    ratio_scheduler = scheduler_factory(config['scheduler_type'])
    strategy = strategy_factory(config['strategy_type'])

import time
start_time = time.time()
for epoch in range(1, 1 + config["epochs"]):
  data_loader = train_loader
  if sampling_flag:
    sampling_ratio = ratio_scheduler(epoch, config['epochs'], config['start_ratio'])
    mt_flag, mt_intervals = strategy(epoch, config['epochs'], startup_ratio=config['start_ratio'])
    if sampling_ratio and float(sampling_ratio) == 1.0:
      data_loader = train_loader
    else:
        if mt_flag:  # i.e., resampling with a new sampling_ratio
            task_type = 'graph_classification'
            rocauc_flag = True
            data_loader = grant_sampler(
                model, train_loader, config['batch_size'], transform,
                config['target_index'], sampling_ratio, config['sampling_type'], task_type, rocauc_flag, device)
        elif not mt_flag and mt_intervals is None:  # i.e., no mt_flag at all, just use the train_loader.
            data_loader = train_loader
        else:  # 1) not mt_flag & 2) mt_intervals is not None. use last_data_loader.
            data_loader = last_data_loader
        last_data_loader = data_loader
    train_total_len = len(data_loader.dataset)
    logger.info(f'epoch: {epoch}, sampling_ratio: {sampling_ratio}, mt_flag: {mt_flag}, train_total_len: {train_total_len}')  
  if epoch == 10:
    # reduce learning rate at this point
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']*0.5, weight_decay=config['weight_decay'])
  loss = train(model, device, data_loader, optimizer, loss_fn)
  train_perf = eval(model, device, data_loader, evaluator)
  val_perf = eval(model, device, valid_loader, evaluator)
  if not args.hide_test_metric:
      # not necessary as output unused during train loop but needed for reproduciblility as affects number of random number generations, affecting ability to generate previously observed outputs depending on seed
      test_perf = eval(model, device, test_loader, evaluator)
  train_metric, valid_metric = train_perf[dataset.eval_metric], val_perf[dataset.eval_metric]
  test_metric = test_perf[dataset.eval_metric]
  if valid_metric >= best_valid_metric_at_save_checkpoint and train_metric >= best_train_metric_at_save_checkpoint:
    logger.info(f"New best validation score: {valid_metric} ({dataset.eval_metric}) without training score regression")
    best_valid_metric_at_save_checkpoint = valid_metric
    best_train_metric_at_save_checkpoint = train_metric
    best_model = copy.deepcopy(model)
  logger.info(f'Dataset {config["dataset_id"]}, Epoch: {epoch}, Train: {train_metric:.6f} ({dataset.eval_metric}), Valid: {valid_metric:.6f} ({dataset.eval_metric}), Test: {test_metric:.6f} ({dataset.eval_metric})')
    # no need to display test performance during training steps even if this is a deterministic seed used for reporting test performance (post-hyperparameter selection)
    # the test would already not be used for selection but no reason to even log it more than once at the end
    # (we can just report it once the training finishes for this same run below)
    # (this particular script was originally written just to report the completed model and reproduce the leaderboard result, not for hyperparameter tuning)
logger.info(f'total_time: {time.time() - start_time}')

with open(f"best_{config['dataset_id']}_gin_model_{config['num_layers']}_layers_{config['hidden_dim']}_hidden.pkl", "wb") as f:
  pickle.dump(best_model, f)

train_metric = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
valid_metric = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_valid")[dataset.eval_metric]
test_metric = None
if not args.hide_test_metric:
    test_metric  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_filename=f"gin_{config['dataset_id']}_test")[dataset.eval_metric]

logger.info(f'Best model for {config["dataset_id"]} (eval metric {dataset.eval_metric}): Train: {train_metric:.6f}, Valid: {valid_metric:.6f}, Test: {test_metric:.6f}')
logger.info(f"parameter count: {sum(p.numel() for p in best_model.parameters())}")
