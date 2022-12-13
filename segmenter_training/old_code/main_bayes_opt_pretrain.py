from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
import numpy as np
from glob import glob
import os
import json
from shutil import rmtree
from bayes_opt.util import load_logs as load_logs_bayes
from bayes_opt import UtilityFunction as UtilityFunction_bayes

from config import Config
from train import train
from optimize_segmentation import optimize_segmentation
from test_fcn import test_fcn
from split_train_test import split_train_test



def train_one_model(pretrain_num_blocks,pretrain_max_block_size,pretrain_noise_std_fraction,
                    pretrain_noise_pixel_p,pretrain_chessboard_num_blocks,pretrain_chessboard_max_block_size,
                    pretrain_rot_num_blocks,pretrain_rot_max_block_size,iter_,pret_num):
    
    
    pretrain_num_blocks = int(pretrain_num_blocks)
    pretrain_max_block_size = int(pretrain_max_block_size)
    pretrain_chessboard_num_blocks = int(pretrain_chessboard_num_blocks)
    pretrain_chessboard_max_block_size = int(pretrain_chessboard_max_block_size)
    pretrain_rot_num_blocks = int(pretrain_rot_num_blocks)
    pretrain_rot_max_block_size = int(pretrain_rot_max_block_size)

    
    
    
    results = {'mean_jacard':[],'mean_binary_jacard':[],'aps':[],'segmenter_name':[],'method':[],'border_width':[],
                   'pretraind_model_name':[],'cell_type_train':[],'cell_type_opt':[],'cell_type_res':[],'mean_valid_jacard':[],
                   'pretrain_num_blocks':[],'pretrain_max_block_size':[],'pretrain_noise_std_fraction':[],'pretrain_noise_pixel_p':[],
                   'pretrain_chessboard_num_blocks':[],'pretrain_chessboard_max_block_size':[],'pretrain_rot_num_blocks':[],
                   'pretrain_rot_max_block_size':[]}
    
    
    config = Config()
    
    if os.path.isdir(config.model_save_dir):
        rmtree(config.model_save_dir) 
    
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)
    
    

    names_train =  glob(config.data_pretrain_train_valid_path+ os.sep+'train' + '/**/*.tif',recursive=True)
    names_valid = glob(config.data_pretrain_train_valid_path+ os.sep+'valid'+'/**/*.tif',recursive=True)

    
    config.model_name_load = None
    config.method = 'pretraining'
    config.border_width = 88
    
    
    
    config.pretrain_num_blocks = pretrain_num_blocks
    config.pretrain_max_block_size = pretrain_max_block_size
    config.pretrain_noise_std_fraction = pretrain_noise_std_fraction
    config.pretrain_noise_pixel_p = pretrain_noise_pixel_p
    config.pretrain_chessboard_num_blocks = pretrain_chessboard_num_blocks
    config.pretrain_chessboard_max_block_size = pretrain_chessboard_max_block_size
    config.pretrain_rot_num_blocks = pretrain_rot_num_blocks
    config.pretrain_rot_max_block_size = pretrain_rot_max_block_size
    
    pretraind_model_name = train(config,names_train,names_valid)
    
    cell_type_train = '*'
    cell_type_opt = '*'
    cell_type_res = '*'

    config.model_name_load = pretraind_model_name
    config.method = 'semantic'
    config.border_width =10

    names_train =  glob(config.data_train_valid_test_path+ os.sep+'train' +'/**/*' + cell_type_train + '_img.tif', recursive=True)
    names_valid = glob(config.data_train_valid_test_path+ os.sep+'valid'+'/**/*' + cell_type_train + '_img.tif', recursive=True)
    fg_model_name = train(config,names_train,names_valid)


    config.method = 'dt'
    # config.method = 'boundary_line'
    config.border_width = 4
    config.model_name_load = pretraind_model_name
    
    names_train =  glob(config.data_train_valid_test_path+ os.sep+'train' +'/**/*' + cell_type_train + '_img.tif', recursive=True)
    names_valid = glob(config.data_train_valid_test_path+ os.sep+'valid'+'/**/*' + cell_type_train + '_img.tif', recursive=True)
    best_model_name = train(config,names_train,names_valid)

    names_valid =  glob(config.data_train_valid_test_path+ os.sep+'valid' +'/**/*' + cell_type_opt + '_img.tif', recursive=True)
    segmenter_name,value = optimize_segmentation(config,fg_model_name,best_model_name,names_valid,get_value=True)
    
    
    names_test =  glob(config.data_train_valid_test_path + os.sep + 'test' +'/**/*' + cell_type_res + '_img.tif', recursive=True)
                        
    mean_jacard,mean_binary_jacard,aps = test_fcn(config,segmenter_name,names_test)
    
    
    results['mean_jacard'].append(mean_jacard)
    results['mean_binary_jacard'].append(mean_binary_jacard)
    results['aps'].append(aps)
    results['segmenter_name'].append(segmenter_name)
    results['method'].append(config.method)
    results['border_width'].append(config.border_width)
    results['pretraind_model_name'].append(pretraind_model_name)
    results['cell_type_train'].append(cell_type_train)
    results['cell_type_opt'].append(cell_type_opt)
    results['cell_type_res'].append(cell_type_res)
    results['mean_valid_jacard'].append(value)
    results['pretrain_num_blocks'].append(pretrain_num_blocks)
    results['pretrain_max_block_size'].append(pretrain_max_block_size)
    results['pretrain_noise_std_fraction'].append(pretrain_noise_std_fraction)
    results['pretrain_noise_pixel_p'].append(pretrain_noise_pixel_p)
    results['pretrain_chessboard_num_blocks'].append(pretrain_chessboard_num_blocks)
    results['pretrain_chessboard_max_block_size'].append(pretrain_chessboard_max_block_size)
    results['pretrain_rot_num_blocks'].append(pretrain_rot_num_blocks)
    results['pretrain_rot_max_block_size'].append(pretrain_rot_max_block_size)
    
    # with open('../result_opt_' + str(iter_) + '.json', 'w') as outfile:
    with open('../result_all_pret_' + str(iter_) + '_' + str(pret_num) + '.json', 'w') as outfile:
            json.dump(results, outfile) 
    
    
    return value


class Wrapper(object):
    def __init__(self,iter_init,pret_num):
        self.iter= iter_init
        self.pret_num = pret_num

    def __call__(self, **params_in):
        
        
        params = dict()
        params['pretrain_num_blocks'] = 0
        params['pretrain_max_block_size'] = 99
        params['pretrain_noise_std_fraction'] = 0
        params['pretrain_noise_pixel_p'] = 0
        params['pretrain_chessboard_num_blocks'] = 0
        params['pretrain_chessboard_max_block_size'] = 99
        params['pretrain_rot_num_blocks'] = 0
        params['pretrain_rot_max_block_size'] = 99
        
        for key in list(params_in.keys()):
            
            params[key] = params_in[key]

        
        
        self.iter = self.iter + 1
        return train_one_model(params['pretrain_num_blocks'],params['pretrain_max_block_size'],
                               params['pretrain_noise_std_fraction'],params['pretrain_noise_pixel_p'],
                               params['pretrain_chessboard_num_blocks'],params['pretrain_chessboard_max_block_size'],
                               params['pretrain_rot_num_blocks'],params['pretrain_rot_max_block_size'], self.iter,self.pret_num )



if __name__ == "__main__":
    
    logging.basicConfig(filename='debug.log',level=logging.INFO)

    
    try:
    # if True:
        
        pbounds_all = []
        
        
        pbounds = {'pretrain_num_blocks':[0,50],
                    'pretrain_max_block_size':[10,70],
                    }
        pbounds_all.append(pbounds)
        
        pbounds = {'pretrain_noise_std_fraction':[0,1],
                    'pretrain_noise_pixel_p':[0,0.5],
                    }
        pbounds_all.append(pbounds)
        
        pbounds = {'pretrain_chessboard_num_blocks':[0,50],
                    'pretrain_chessboard_max_block_size':[10,70],
                    }
        pbounds_all.append(pbounds)
        
        
        pbounds = {'pretrain_rot_num_blocks':[0,50],
                    'pretrain_rot_max_block_size':[10,70],
                    }
        pbounds_all.append(pbounds)
        
        
        
        
        for pret_num,pbounds in enumerate(pbounds_all):

        
            config = Config()
            
            if not os.path.isdir(config.data_train_valid_test_path):
                split_train_test()
                
            if not os.path.isdir(config.best_models_dir):
                os.mkdir(config.best_models_dir)
                
            if not os.path.isdir(config.model_save_dir):
                os.mkdir(config.model_save_dir)
                
            if not os.path.isdir(config.opt_folder):
                os.mkdir(config.opt_folder)
                
                
            
            optimizer_bayes = BayesianOptimization(f=Wrapper(0,pret_num),pbounds=pbounds,random_state=0)  
            
            # load_logs_bayes(optimizer_bayes, logs=['../opt_resbayes_opt_log.json']);
            
            # logger_bayes = JSONLogger(path= '../opt_resbayes_opt_log.json')
            logger_bayes = JSONLogger(path= '../opt_all_pret_' + str(pret_num) + '.json')
            optimizer_bayes.subscribe(Events.OPTIMIZATION_STEP, logger_bayes)
            
            
            
    
            optimizer_bayes.maximize(init_points=2,n_iter=8)

    except Exception as e:
        logging.critical(e, exc_info=True)









