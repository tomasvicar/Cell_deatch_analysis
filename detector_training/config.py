import numpy as np
import os

class Config:
    
    data_dir = ''
    save_dir = ''
    best_models_dir = ''
    

    split_ratio_train_valid_test=[8,1,1]
     
    train_batch_size = 16
    train_num_workers = 8
    valid_batch_size = 4
    valid_num_workers = 2


    init_lr = 0.001
    lr_changes_list = np.cumsum([700, 100, 50])
    gamma = 0.1
    max_epochs = lr_changes_list[-1]
    
    

    filters=list((np.array([64,128,256,512,1024])/4).astype(int))
    in_size=1
    out_size=1
    
    
    device='cuda:0'
    
    patch_size=256
    
    sigma = 2.5
    div = 0.04897406782784436
    
    too_close = 2
    
    norm_dapi = [50, 600]
    
    norm_qpi = [-3, 9]
    

    
    
    