# !/bin/bash

target_prop="mu" #Available options: homo, lumo, alpha
device_no=2 #gpu number
expt_no=1
data_set="qm9_pyg_25k_0.pt" #processed dataset.  Should be saved in dataset/qm9/processed folder
selection_method="unc_div" #, random, lloss,coreset, unc_div
epochs=750 #number of epochs to train
backbone="dimenetpp" #dimenetpp, spherenet
addendum=1500 #number of samples to label in each AL iteration
init_size=5000 #number of initially labeled samples
train_size=25000 #total number of samples in data pool (labeled + unlabeled)
valid_size=10000 #number of samples in valid size
# for cycle_no in {0} #No of AL iterations
# do
    
# done
source ~/.bashrc && conda init && conda activate grant && python3 train.py --init_size $init_size --train_size $train_size --valid_size $valid_size \
    --ADDENDUM $addendum --backbone $backbone \
    --epochs $epochs --device $device_no --target $target_prop --dataset $data_set \
    --selection_method $selection_method --cycle 0 --expt $expt_no
