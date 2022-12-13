from glob import glob
from os import makedirs
import os
import pickle
import torch
from skimage.io import imread,imsave
import numpy as np
import matplotlib.pyplot as plt

data_path = r'D:\qpi_segmentation_tmp\clasical_dt'


for it in range(5):

    for split in ['valid','test']:

        data_to_predict_valid = glob(data_path + '/data_train_valid_test' + str(it) + '/' + split + '/**/*_img.tif',recursive=True )
        
        
        
        
        for name in data_to_predict_valid:
            
            
            cell_type = name.split('_')[-2]
            
            tmp = data_path + '/predicted_fg' + str(it) + '/' + split + '/' + cell_type
            if not os.path.isdir(tmp):
                makedirs(tmp)
            
            
            name_save = name
            
            name_save = data_path + '/predicted_fg' + str(it) + '/' + split + '/' + cell_type + '/' + os.path.split(name_save)[1].replace('_img.tif','_fg.png')
            
            
            segmenter_name = data_path + '/dt_' + str(it) +  '.p'
            
            segmenter = pickle.load( open( segmenter_name, "rb" ) )
            
            
            
            img=imread(name)
            img0=img.astype(np.float32)
            img=img0.copy()
            
        
            
            shape=np.shape(img)
            img=torch.from_numpy(img.reshape((1,1,shape[0],shape[1])).astype(np.float32))
        
            
            
            
            dt,seg = segmenter.predict_imgs(img)
            
            imsave(name_save,seg)
            
            
            
            
            
            
            
            
            
        
        
        











