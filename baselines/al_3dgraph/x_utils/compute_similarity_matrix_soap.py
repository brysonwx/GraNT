import argparse
import io
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from fastdist import fastdist
from ase.io import read, write
from dscribe.descriptors import SOAP

from torch_geometric.loader import DataLoader

from datasets import QM93D
from al import SubsetSequentialSampler


def get_split_idx(train_size, np_file):
    indices = np.arange(train_size)
    labeled_set = np.load(f"{np_file}")
    unlabeled_set = np.setdiff1d(indices, labeled_set)
    
    assert np.intersect1d(labeled_set, unlabeled_set).size == 0 
    assert np.union1d(labeled_set, unlabeled_set).size == train_size
    return {'lab':labeled_set.tolist(), 'unlab': unlabeled_set.tolist()}

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def pairwise_usr_similarity(matrix1, matrix2=None):
    if matrix2 is None:
        matrix2 = matrix1
        
    # Compute distances
    distances = torch.cdist(matrix1, matrix2, p=1) / 4.0 + 1.0
    similarities = 1.0 / distances
    return similarities


def similarity_soap(opt):
    device = opt.device
    batch_size = opt.batch_size
    dataset = opt.dataset
    selection_method = opt.selection_method
    init_size = opt.init_size
    train_size = opt.train_size
    valid_size = opt.valid_size
    
    dataset = QM93D(root='baseline/al_3dgraph/x_dataset/', processed_fn=dataset)
    
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=train_size,valid_size=valid_size, 
                                      init_size=init_size, seed=42, method= selection_method)

    train_dataset, valid_dataset, test_dataset, unlab_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]

    unlab_loader =DataLoader(dataset, batch_size=1, 
                            sampler=SubsetSequentialSampler(split_idx['unlabeled']), # more convenient if we maintain the order of subset
                            pin_memory=True, drop_last=False)
    print(len(unlab_loader))
    
    #######
    #######
    
    ele_no = {
        1: 'H',  
        6: 'C',  
        7: 'N',  
        8: 'O',  
        9: 'F',  
    }

    atoms_all = []
    for i, batch in enumerate(unlab_loader):
        res = ''
        
        atoms = batch['z'].tolist()
        res +=str(len(atoms)) + '\n' + '\n'
        coord = batch['pos'].tolist()
        
        for atom_n, coord in zip(atoms, coord):

            temp = ele_no[atom_n]+ '\t' + '\t'.join(map(str, coord))+'\n'
            res +=temp
            
        atoms_all.append(read(io.StringIO(res), format='xyz'))
        print(i)
        # SOAP parameters
    print(len(atoms_all))
    species = ["C", "H", "O", "N", "F"]
    r_cut = 6
    n_max = 10
    l_max = 6

    # setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
    )

    soap_mol = soap.create(atoms_all)


    with torch.cuda.device(device):
        features1 = torch.tensor([]).to(device)
    for i,sm in enumerate(tqdm(soap_mol)):
        a =torch.tensor(sm).to(device)
        b  = a.mean(axis=0, keepdim=True)
        features1 = torch.cat((features1, b), 0)
    
    ###########
    ###########


    # lab_predictions = features1[:len(labeled_indices)]
    unlab_predictions = features1
    
    similarity_matrix = pairwise_usr_similarity(unlab_predictions)
    similarity_matrix = similarity_matrix/similarity_matrix.max()
    similarity_matrix.fill_diagonal_(30.0)
    similarity_matrix = similarity_matrix.cpu()
    return similarity_matrix