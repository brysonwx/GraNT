
import time
import os
import torch
from torch.optim import Adam, SGD
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import sys
from . import get_average_node_per_molecule
# from .spherenet.ConcreteDropout import ConcreteDropout

class run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass
        
    def run(self, device, train_dataset, valid_dataset, test_dataset, model,
        loss_func, evaluation, method='random',lossnet = None, lossnet_loss_func = None,
        epochs=500, batch_size=32, vt_batch_size=32, lr=0.0005, 
        lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        energy_and_force=False, p=100, save_dir='', log_dir='', expt=1, cycle=0):
        r"""
        The run script for training and validation.
        
        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function. 
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)
        
        """        

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

       
       #define optimizer and scheduler for lossnet here
        
        optimizer_lossnet = None 
        scheduler_lossnet = None
        if method == 'lloss':
            lossnet = lossnet.to(device)
            num_params = sum(p.numel() for p in lossnet.parameters())
            print(f'#Params lossnet: {num_params}')
            optimizer_lossnet = Adam(lossnet.parameters(), lr=5e-4, weight_decay=weight_decay)
            scheduler_lossnet = StepLR(optimizer_lossnet, step_size=25, gamma=lr_decay_factor)

        ###################

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')
        best_test_only = float('inf')
        
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
            
        #print("Initial weights: ", model.state_dict())
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            print("\n=====Epoch {}".format(epoch), flush=True)
            
            print('\nTraining...', flush=True)
            train_mae = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device, method, lossnet, lossnet_loss_func, optimizer_lossnet)

            print('\n\nEvaluating...', flush=True)
            valid_mae = self.val(model, valid_loader, energy_and_force, p, evaluation, device, method)

            print('\n\nTesting...', flush=True)
            test_mae = self.val(model, test_loader, energy_and_force, p, evaluation, device, method)

            print({'Epoch': epoch, 'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})

            if log_dir != '':
                writer.add_scalar('train', train_mae, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('test_mae', test_mae, epoch)
            
            if test_mae < best_test_only:
                best_test_only = test_mae
            
            if valid_mae < best_valid:
                best_valid = valid_mae
                best_test = test_mae
                if save_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            scheduler.step()
            if method == 'lloss':
                scheduler_lossnet.step()
   

        print(f'Best validation MAE so far: {best_valid}')
        print(f'Test MAE when got best validation result: {best_test}')
        end_time = time.time()
        print(f'total time: {end_time - start_time}')

        if not os.path.exists(f'runs/{method}/run{expt}/res.txt'):
            with open(os.path.join(f'runs/{method}/run{expt}', f'res.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle}  total_time: {end_time - start_time}  BEST:: valid:{best_valid} test_v: {best_test} test_o:{best_test_only}\n')
                
        with open(os.path.join(f'runs/{method}/run{expt}', f'res.txt'), 'a') as f:
            f.write(f'CYCLE: {cycle}   total_time: {end_time - start_time} BEST:: valid:{best_valid} test_v: {best_test} test:{best_test_only}\n')

        
        # torch.save(model.state_dict(), os.path.join(f'runs/{method}/run{expt}', f'model_{cycle}.pth'))

        if log_dir != '':
            writer.close()

    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device, 
            method, lossnet, lossnet_loss_func, optimizer_lossnet):
        r"""
        The script for training.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training. 
            device (torch.device): The device where the model is deployed.

        :rtype: Traning loss. ( :obj:`mae`)
        
        """   
        model.train()
        #disable_dropout(model)
        loss_accum = 0
        lossnet_loss_accum = 0
        
        if method == 'lloss':
            lossnet.train()
            
        
        for step, batch_data in enumerate(tqdm(train_loader)):
            
            #zero grad optimizers
            optimizer.zero_grad()
            
            if method == 'lloss':
                optimizer_lossnet.zero_grad()

            batch_data = batch_data.to(device)
            if method == 'unc_div':
                edge_index, feature_dict, out, regularization = model(batch_data)
            else:
                edge_index, feature_dict, out = model(batch_data)
            #print(regularization)
            #sys.exit()
            edge_features = feature_dict['edge_features']
            node_features = feature_dict['node_features']


        
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                f_loss = loss_func(force, batch_data.force)
                loss = e_loss + p * f_loss
            else:
          
                loss = loss_func(out, batch_data.y.unsqueeze(1)) 
                
            total_loss = loss
            #training for lossnet
            if method == 'lloss':
                features_list = []


                for n_feature in node_features:
                    av_node_features = get_average_node_per_molecule(batch_data, n_feature.detach(), device)
                    features_list.append(av_node_features)
               
                lossnet_out = lossnet(features_list)
                lossnet_out = lossnet_out.view(lossnet_out.size(0))
                lossnet_loss = lossnet_loss_func(lossnet_out, torch.abs(out - batch_data.y.unsqueeze(1)).view(out.size(0)))
                total_loss =  total_loss + 0.001 * lossnet_loss
                # total_loss =  total_loss + 0.001 * lossnet_loss
                # print('Loss after adding lossnet loss: ', loss)
                
            total_loss.backward(retain_graph=True)
            # print(model.update_es[0].layers_before_skip[0].lin2.p_logit.grad)
            optimizer.step()
            if method == 'lloss':
                # print('lossnet optimizer')

                optimizer_lossnet.step()
                lossnet_loss_accum += lossnet_loss.detach().cpu().item()

            loss_accum += loss.detach().cpu().item()
        return (loss_accum / (step + 1), lossnet_loss_accum/ (step + 1))

    def val(self, model, data_loader, energy_and_force, p, evaluation, device, method):
        r"""
        The script for validation/test.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)
        
        """   
        model.eval()
        # disable_dropouta(model)

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device)
            targets_force = torch.Tensor([]).to(device)
        
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            if method == 'unc_div':
                _, _, out, _ = model(batch_data)
            else:
                _, _, out = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                preds_force = torch.cat([preds_force,force.detach_()], dim=0)
                targets_force = torch.cat([targets_force,batch_data.force], dim=0)
            preds = torch.cat([preds, out.detach_()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

        input_dict = {"y_true": targets, "y_pred": preds}

        if energy_and_force:
            input_dict_force = {"y_true": targets_force, "y_pred": preds_force}
            energy_mae = evaluation.eval(input_dict)['mae']
            force_mae = evaluation.eval(input_dict_force)['mae']
            print({'Energy MAE': energy_mae, 'Force MAE': force_mae})
            return energy_mae + p * force_mae

        return evaluation.eval(input_dict)['mae']