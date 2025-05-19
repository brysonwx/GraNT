import os.path as osp
import os
from typing import Any, Callable, List, Optional, Tuple, Union
from collections.abc import Sequence
import random


import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url, Data
# from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import BaseData




IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class QM93D(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset 
        which is from `"Quantum chemistry structures and properties of 134 kilo molecules" <https://www.nature.com/articles/sdata201422>`_ paper.
        It connsists of about 130,000 equilibrium molecules with 12 regression targets: 
        :obj:`mu`, :obj:`alpha`, :obj:`homo`, :obj:`lumo`, :obj:`gap`, :obj:`r2`, :obj:`zpve`, :obj:`U0`, :obj:`U`, :obj:`H`, :obj:`G`, :obj:`Cv`.
        Each molecule includes complete spatial information for the single low energy conformation of the atoms in the molecule.

        .. note::
            We used the processed data in `DimeNet <https://github.com/klicperajo/dimenet/tree/master/data>`_, wihch includes spatial information and type for each atom.
            You can also use `QM9 in Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9>`_.

    
        Args:
            root (string): the dataset folder will be located at root/qm9.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = QM93D()
        >>> target = 'mu'
        >>> dataset.data.y = dataset.data[target]
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(Cv=[32], G=[32], H=[32], U=[32], U0=[32], alpha=[32], batch=[579], gap=[32], homo=[32], lumo=[32], mu=[32], pos=[579, 3], ptr=[33], r2=[32], y=[32], z=[579], zpve=[32])

        Where the attributes of the output data indicates:
    
        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The target property for the graph (molecule).
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    """
    def __init__(self, root = 'x_dataset/', transform = None, pre_transform = None, pre_filter = None, processed_fn='qm9_pyg_25k_0.pt'):
        self.processed_fn = processed_fn
        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'qm9')

        super(QM93D, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__
        

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return self.processed_fn

    def download(self):
        download_url(self.url, self.raw_dir)

    # def process(self):
        
        # data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        # R = data['R']
        # Z = data['Z']
        # N= data['N']
        # split = np.cumsum(N)
        # R_qm9 = np.split(R, split)
        # Z_qm9 = np.split(Z,split)
        # target = {}
        # for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']:
        #     target[name] = np.expand_dims(data[name],axis=-1)
        # # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)

        # data_list = []
        # for i in tqdm(range(len(N))):
        #     R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
        #     z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
        #     y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']]
        #     data = Data(pos=R_i, z=z_i, y=y_i[0], mu=y_i[0], alpha=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4], r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11])

        #     data_list.append(data)

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # print(data_list[0:2])

        # expt = 1
        # val_set = np.load(f"runs/archive/run{expt}/val_set.npy").tolist()
        # test_set = np.load(f"runs/archive/run{expt}/test_set.npy").tolist()
        # init_set =  np.load(f"runs/archive/run{expt}/init_5k_ori.npy").tolist()

        # sel = val_set + test_set + init_set

        # u_s = [i for i in range(len(data_list)) if i not in sel]

        # assert sorted(sel+u_s) == list(range(len(data_list)) )
        # val_data = [data_list[i] for i in val_set]
        # test_data = [data_list[i] for i in test_set]
        # init_data = [data_list[i] for i in init_set]
        # rem_dta = [data_list[i] for i in u_s]



        # print('lengths: ', len(init_data), len(val_data), len(test_data), len(rem_dta))

        # for i in range(3):
        #     kk = shuffle(rem_dta, random_state=i ** 5)
        #     flfful = kk[:20000]
        #     dl = init_data + flfful + val_data + test_data
        #     print(dl[:1])
        #     print(val_data[:1])
        #     print(test_data[:1])
        #     print(type(dl))
        #     print(len(dl))
        #     data, slices = self.collate(dl)
        #     print('Saving...')
        #     torch.save((data, slices), f'dataset/qm9/processed/qm9_pyg_25k_{i}.pt')
        

        
        # torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed, method = 'random', cycle=0, expt=1, ADDENDUM=1500, init_size=5000):
        
        indices = np.arange(train_size)
        labeled_set = np.load(f"runs/{method}/run{expt}/init_set.npy")
        unlabeled_set = np.setdiff1d(indices, labeled_set)
  
        assert len(np.intersect1d(labeled_set,unlabeled_set)) == 0
        labeled_set = labeled_set.tolist()
        unlabeled_set = unlabeled_set.tolist()

        print('First assert passed...')
        print('Unlab len: ', len(unlabeled_set))
        print('train len: ', len(labeled_set))


       
        print('Len labeled:', len(labeled_set))
        assert len(labeled_set) == (cycle * ADDENDUM + init_size)
    
        train_idx, val_idx, test_idx, unlabeled_index = torch.tensor(labeled_set), torch.tensor(list(range(train_size,train_size+valid_size))), torch.tensor(list(range(train_size+valid_size, data_size))), torch.tensor(unlabeled_set)
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx, 'unlabeled': unlabeled_index}
        return split_dict
        
        
       

if __name__ == '__main__':

    dataset = QM93D(processed_fn='qm9_pyg_25k_0.pt')
    print(dataset[1])
    