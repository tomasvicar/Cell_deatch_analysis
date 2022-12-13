import logging
import sys
import numpy as np
from glob import glob
import os

from config import Config
from train import train
from optimize_segmentation import optimize_segmentation
from test_fcn import test_fcn
from split_train_test import split_train_test


if __name__ == "__main__":
    
    logging.basicConfig(filename='debug.log',level=logging.INFO)
    # try:
    
    if True:
        
        config = Config()
        
        if not os.path.isdir(config.data_train_valid_test_path):
            split_train_test()
            
        if not os.path.isdir(config.best_models_dir):
            os.mkdir(config.best_models_dir)
            
        if not os.path.isdir(config.model_save_dir):
            os.mkdir(config.model_save_dir)

        
        
        
        
        names_train =  glob(config.data_pretrain_train_valid_path+ os.sep+'train' + '/**/*.tif',recursive=True)
        names_valid = glob(config.data_pretrain_train_valid_path+ os.sep+'valid'+'/**/*.tif',recursive=True)

        
        config.model_name_load = None
        config.method = 'pretraining'
        config.border_width = 88
        
        config.pretrain_num_blocks = 20
        config.pretrain_max_block_size = 50
        pretraind_model_name_tmp1 = train(config,names_train,names_valid)

        config.pretrain_num_blocks = 30
        config.pretrain_max_block_size = 60
        pretraind_model_name_tmp2 = train(config,names_train,names_valid)
        
        config.pretrain_num_blocks = 15
        config.pretrain_max_block_size = 40
        pretraind_model_name_tmp3 = train(config,names_train,names_valid)

        

        for pretraind_model_name in ['imagenet']:
        
            config.model_name_load = pretraind_model_name
            config.method = 'semantic'
            config.border_width =10
            
            
            
            names_train =  glob(config.data_train_valid_test_path+ os.sep+'train' +'/**/*_img.tif', recursive=True)
            names_valid = glob(config.data_train_valid_test_path+ os.sep+'valid'+'/**/*_img.tif', recursive=True)
            fg_model_name = train(config,names_train,names_valid)
            # fg_model_name = '../best_models/semantic_65_0.00100_gpu_0.00000_train_0.07348_valid_0.04917.pt'
    
            settings = [['dt',10]]
            
    
            
            jacards = []
            segmenter_names = [] 
            
            for model_num,setting in enumerate(settings):
                
                config.method = setting[0]
                config.border_width = setting[1]
                config.model_name_load = pretraind_model_name
                
                names_train = glob(config.data_train_valid_test_path+ os.sep+'train'+'/**/*_img.tif', recursive=True)
                names_valid = glob(config.data_train_valid_test_path+ os.sep+'valid'+'/**/*_img.tif', recursive=True)
                best_model_name = train(config,names_train,names_valid)
                # best_model_name = '../best_models/dt_84_0.00010_gpu_0.00000_train_0.00658_valid_0.00384.pt'
                
                print(best_model_name)
                
                names_valid =  glob(config.data_train_valid_test_path+ os.sep+'valid'+'/**/*_img.tif',recursive=True)
                segmenter_name = optimize_segmentation(config,fg_model_name,best_model_name,names_valid)
                
                print(segmenter_name)
                
                names_test =  glob(config.data_train_valid_test_path+ os.sep+'test'+'/**/*_img.tif',recursive=True)
                mean_jacard = test_fcn(config,segmenter_name,names_test)
                
                
                segmenter_names.append(segmenter_name)
                jacards.append(mean_jacard)
                
            np.save('res.npy',jacards) 
            np.save('names.npy',segmenter_names) 
            
    # except Exception as e:
    #     logging.critical(e, exc_info=True)
        
        
        
        
        
        
        
        
        
