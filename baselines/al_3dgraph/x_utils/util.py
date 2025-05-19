import torch
import torch.nn as nn
import torch.nn.functional as F

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

class MarginRankingLoss_learning_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginRankingLoss_learning_loss, self).__init__()
        self.margin = margin
    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        pred_loss = inputs[random]
        pred_lossi = inputs[:inputs.size(0)//2]
        pred_lossj = inputs[inputs.size(0)//2:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:inputs.size(0)//2]
        target_lossj = target_loss[inputs.size(0)//2:]
        final_target = torch.sign(target_lossi - target_lossj)
        
        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin, reduction='mean')

def get_average_edge_per_molecule(batch_data, edge_index, edges_features, device):
    unique_molecules, nodes_per_molecule = batch_data.batch.unique(return_counts=True)
    maximum_node_index_per_molecule = torch.cumsum(nodes_per_molecule,0)
    maximum_node_index_per_molecule = [0] + maximum_node_index_per_molecule.tolist() #0 for using conditional
    

    # print('maximum_node_index_per_molecule: ',maximum_node_index_per_molecule)
    num_edges_per_molecule = []
    for i in range(1, len(maximum_node_index_per_molecule)):
        edges_per_mol = edge_index[(edge_index>=maximum_node_index_per_molecule[i-1]) & (edge_index<maximum_node_index_per_molecule[i])]
        # print(f'max node index per mol: {edges_per_mol.max()}')

        num_edges_per_molecule.append(edges_per_mol.size(0))


    #spliting edges per molecule.
    edges_split = torch.split(edges_features, num_edges_per_molecule, dim=0)

    features = torch.tensor([]).to(device)
    for one_mol_edges in edges_split:
        # print('one_mol_edges.shape: ', one_mol_edges.shape)
        mean_edge = torch.mean(one_mol_edges, dim=0, keepdim=True)
        # print(mean_edge.shape)
        features = torch.cat((features, mean_edge), 0)
    return features


def get_average_edge_node_per_molecule(batch_data, edge_index, edges_features, nodes_features, device):
    unique_molecules, nodes_per_molecule = batch_data.batch.unique(return_counts=True)
    maximum_node_index_per_molecule = torch.cumsum(nodes_per_molecule,0)
    maximum_node_index_per_molecule = [0] + maximum_node_index_per_molecule.tolist() #0 for using conditional

    num_edges_per_molecule = []
    for i in range(1, len(maximum_node_index_per_molecule)):
        edges_per_mol = edge_index[(edge_index>=maximum_node_index_per_molecule[i-1]) & (edge_index<maximum_node_index_per_molecule[i])]

        num_edges_per_molecule.append(edges_per_mol.size(0))


    #spliting edges per molecule.
    edges_split = torch.split(edges_features, num_edges_per_molecule, dim=0)
    ####
    nodes_split = torch.split(nodes_features, nodes_per_molecule.tolist(), dim=0)

    features = torch.tensor([]).to(device)
    for one_mol_edges, one_mol_nodes in zip(edges_split, nodes_split):
        mean_edge = torch.mean(one_mol_edges, dim=0, keepdim=True)
        mean_node = torch.mean(one_mol_nodes, dim=0, keepdim=True)
        mean_node = mean_node.repeat(1,8)
        mean_edge_node = torch.cat((mean_edge, mean_node), 1)
        features = torch.cat((features, mean_edge_node), 0)
    return features

def get_average_node_per_molecule(batch_data, nodes_features, device):
    unique_molecules, nodes_per_molecule = batch_data.batch.unique(return_counts=True)
    nodes_split = torch.split(nodes_features, nodes_per_molecule.tolist(), dim=0)

    features = torch.tensor([]).to(device)
    for one_mol_nodes in nodes_split:
        mean_node = torch.mean(one_mol_nodes, dim=0, keepdim=True)
        features = torch.cat((features, mean_node), 0)
    return features