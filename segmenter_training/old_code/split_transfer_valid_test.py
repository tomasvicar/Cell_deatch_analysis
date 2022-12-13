import numpy as np
import os
from shutil import copyfile,rmtree

def split_transfer_valid_test(seed,cell_types,path):
        
    path_save = '../data_transfer_valid_test'
    
    np.random.seed(seed)
    
    try:
        rmtree(path_save)
    except:
        pass
    

    for cell_type in cell_types:

        path_valid = path_save + '/valid/'+ cell_type
        try:
            os.makedirs(path_valid)
        except:
            pass
        
        
        path_test = path_save + '/test/'+ cell_type
        try:
            os.makedirs(path_test)
        except:
            pass

    
    
        names=[]
        for root, dirs, files in os.walk(path + '/' + cell_type):
            for name in files:
                if name.endswith("_img.tif"):
                    name=name.replace('_img.tif','')
                    names.append(root + os.sep +name)
                    
             
        split_nums = [10,15]
             
        perm=np.random.permutation(len(names))   
             
        split_ind=np.array(split_nums)
        split_ind=np.floor(np.cumsum(split_ind/np.sum(split_ind)*len(names))).astype(np.int)
        
        
        valid_ind=perm[:split_ind[0]]
        test_ind=perm[split_ind[0]:]     
                    

          
        for k in valid_ind:
            name_in=names[k]
            name_out=name_in.replace(path + '/' + cell_type, path_valid)
            
            copyfile(name_in + '_img.tif',name_out+'_img.tif')
            copyfile(name_in + '_mask.png',name_out+'_mask.png')
            
        
        for k in test_ind:
            name_in=names[k]
            name_out=name_in.replace(path + '/' + cell_type, path_test)
            
            copyfile(name_in + '_img.tif',name_out+'_img.tif')
            copyfile(name_in + '_mask.png',name_out+'_mask.png')
    
    
    
if __name__ == "__main__":
    
    split_transfer_valid_test(0,['G361','HOB','A2058'])
    
    
    
    
    