import argparse
# import wandb
import torch
import numpy as np
import os
import random
from torch_geometric.loader import DataLoader

from models import LossNet
from datasets import QM93D

from x_utils import *

from al import *

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection_method', default='random', type=str,  help='selection method')
    parser.add_argument('--target', default='mu', type=str,  help='target property')
    parser.add_argument('--backbone', default='spherenet', type=str,  help='backbone model')
    parser.add_argument('--dataset', default='qm9_pyg_25k_0.pt', type=str,  help='.pt dataset')
    parser.add_argument('--device', default=2, type=int,  help='GPU Device')
    parser.add_argument('--cycle', default=0, type=int,  help='AL Cycle')
    parser.add_argument('--expt', default=1, type=int,  help='Eperiment number')
    parser.add_argument('--ADDENDUM', default=1500, type=int,  help='No of data to add')
    parser.add_argument('--epochs', default=750, type=int,  help='No of epochs')
    parser.add_argument('--train_size', default=25000, type=int,  help='total samples in pool (L+U)')
    parser.add_argument('--valid_size', default=10000, type=int,  help='No of samples in valid set')
    parser.add_argument('--init_size', default=5000, type=int,  help='Labeled seed size')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    setup_seed(8848)
    opt = parse_opt()
    selection_method = opt.selection_method
    backbone = opt.backbone
    
    assert selection_method in set(['coreset', 'random', 'lloss', 'dropout','unc_div', 'mcdrop'])
    assert backbone in set(['spherenet', 'dimenetpp'])
    
    if selection_method == 'unc_div':
        if backbone == 'spherenet':
            from models import SphereNet_BGNN as Model
        else:
            from models import DimenetPP_BGNN as Model
            print('Model dimenetpp_bgnn......')
    else:
        if backbone == 'spherenet':
            from models import SphereNet as Model
        else:
            from models import DimeNetPP as Model
    
    print(f'Selection method: {selection_method}')
    device = opt.device
    cycle = opt.cycle
    expt = opt.expt
    ADDENDUM = opt.ADDENDUM
    dataset = opt.dataset
    epochs = opt.epochs
    init_size = opt.init_size
    train_size = opt.train_size
    valid_size = opt.valid_size
    print(f'Dataset: {dataset}')


    dataset = QM93D(root='x_dataset/', processed_fn=dataset)
    target = opt.target # choose from: mu, alpha, homo, lumo, r2, zpve, U0, U, H, G, Cv
    print(f'Target prediction: {target}')
    dataset.data.y = dataset.data[target]

    print(f'Length of entire dataset: {len(dataset.data.y)}')

    split_idx = dataset.get_idx_split(len(dataset.data.y), init_size=init_size, train_size=train_size,valid_size=valid_size, 
                                      seed=42, method= selection_method, cycle=cycle, expt=expt, ADDENDUM=ADDENDUM)

    train_dataset, valid_dataset, test_dataset, unlab_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]
    
    ## training on full dataset##
    # train_dataset, valid_dataset, test_dataset, unlab_dataset = dataset[torch.tensor(split_idx['train'].tolist()+split_idx['unlabeled'].tolist())], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]
    print('train, validaion, test, unlab_dataset:', len(train_dataset), len(valid_dataset), len(test_dataset), len(unlab_dataset))
    print('train: ', train_dataset)
    # print(len(split_idx['train'].tolist()))
    if selection_method == 'unc_div':
        #compute similarity matrix of unlab samples
        
        model = Model(energy_and_force=False, cutoff=5.0, num_layers=4, 
            hidden_channels=128, out_channels=1, int_emb_size=64, 
            basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
            num_spherical=3, num_radial=6, envelope_exponent=5, 
            num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True, droprate=0.2
            )
    else:
        model = Model(energy_and_force=False, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, 
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
        num_spherical=3, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
        )
    model = model.to(device)
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()
 

    lossnet = None
    lossnet_loss_func = None
    if selection_method == 'lloss':
        lossnet = LossNet()
        lossnet = lossnet.to(device)
        lossnet_loss_func = LossPredLoss

    run3d = run()
    run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          method=selection_method,lossnet=lossnet, lossnet_loss_func=lossnet_loss_func,
          epochs=epochs, batch_size=32,
          vt_batch_size=64, lr=0.0005, lr_decay_factor=0.5, 
          lr_decay_step_size=15, expt=expt, cycle=cycle)

    if selection_method == 'random':
        query_indices = random_sel(split_idx['train'].tolist(), split_idx['unlabeled'].tolist(), k=ADDENDUM)
        # print('random len query: ', len(query_indices))
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))


    elif selection_method == 'coreset':
        print('coreset')
        model.eval()
        query_indices, time_taken = coreset(model, split_idx['train'].tolist(), split_idx['unlabeled'].tolist(),dataset[:25000], 
                            device=device, k=ADDENDUM)
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')

   
    ### Our ###
    elif selection_method == 'unc_div':
        print('uncertainty_diversity')
        model.eval()
        query_indices, time_taken = compute_uncertainty_diversity_fast(model, split_idx['train'].tolist(), split_idx['unlabeled'].tolist(),dataset[:25000], 
            device=device, k=ADDENDUM, expt=expt, selection_method='unc_div')
        print(f'Len of next batch:------------  {len(query_indices)}')
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')
                
    
    elif selection_method == 'lloss':
        model.eval()
        lossnet.eval()
        query_indices, time_taken = lloss_sel(model, lossnet, split_idx['unlabeled'].tolist(),dataset[:25000], 
                        device=device, k=ADDENDUM)

        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices.tolist()+split_idx['train'].tolist()))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')