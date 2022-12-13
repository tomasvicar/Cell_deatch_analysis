from glob import glob
import numpy as np

def load_and_split_data(config):
    
    lbls = glob(config.data_path + '/**/*.mat')
    imgs_dapi = [x.replace('_detections.mat', '_DAPI.tiff') for x in lbls]
    imgs_qpi = [x.replace('_detections.mat', '_QPI.tiff') for x in lbls]
    
    
    perm = np.random.permutation(len(lbls))   
         
    split_ind = np.array(config.split_ratio_train_valid_test)
    split_ind = np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(lbls))).astype(np.int)
    
    
    train_ind = perm[:split_ind[0]]
    valid_ind = perm[split_ind[0]:split_ind[1]]         
    test_ind = perm[split_ind[1]:]       
    
    
    names_train = dict()
    names_train['lbls'] = [lbls[x] for x in train_ind]
    names_train['imgs_dapi'] = [imgs_dapi[x] for x in train_ind]
    names_train['imgs_qpi'] = [imgs_qpi[x] for x in train_ind]
    names_valid = dict()
    names_valid['lbls'] = [lbls[x] for x in valid_ind]
    names_valid['imgs_dapi'] = [imgs_dapi[x] for x in valid_ind]
    names_valid['imgs_qpi'] = [imgs_qpi[x] for x in valid_ind]
    names_test = dict()
    names_test['lbls'] = [lbls[x] for x in test_ind]
    names_test['imgs_dapi'] = [imgs_dapi[x] for x in test_ind]
    names_test['imgs_qpi'] = [imgs_qpi[x] for x in test_ind]
    
    names = dict()
    names['names_train'] = names_train
    names['names_valid'] = names_valid
    names['names_test'] = names_test
    
    return names
    


