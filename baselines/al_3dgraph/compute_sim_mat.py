import argparse
import torch
from x_utils import similarity_soap, similarity_our
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=1, type=int,  help='GPU Device')
    parser.add_argument('--batch_size', default=64, type=int,  help='batch size')
    parser.add_argument('--dataset', default='qm9_pyg_25k_0.pt', type=str,  help='.pt dataset')
    parser.add_argument('--descriptor', default='our', type=str,  help='soap or our')
    parser.add_argument('--selection_method', default='unc_div', type=str,  help='soap or our')
    parser.add_argument('--train_size', default=25000, type=int,  help='total samples in pool (L+U)')
    parser.add_argument('--valid_size', default=10000, type=int,  help='No of samples in valid set')
    parser.add_argument('--init_size', default=5000, type=int,  help='Labeled seed size')
    
    # data should be the processed .pt file in al_3dgraph/dataset/processed directory
    return parser.parse_args()

def main(opt):
    
    descriptor_type = opt.descriptor
    if descriptor_type == 'soap':
        similarity_matrix = similarity_soap(opt)
        print(similarity_matrix.shape)
        torch.save(similarity_matrix, 'tensor.pt')
        
    
    elif descriptor_type == 'our':
        similarity_matrix = similarity_our(opt)
        torch.save(similarity_matrix, 'tensor.pt')
    
    else:
        print('Descriptor type not implemented')

if __name__=='__main__':
    opt = parse_opt()
    main(opt)
    
