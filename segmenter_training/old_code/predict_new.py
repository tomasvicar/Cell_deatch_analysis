import torch
import os
from skimage.io import imread,imsave
import numpy as np
import matplotlib.pyplot as plt
from utils.visboundaries import visboundaries
from utils.colorize_notouchingsamecolor import colorize_notouchingsamecolor
import pickle
from utils.get_jacards_cell import get_jacards_cell,get_jacards_cell_with_fp
from glob import glob



save_folder = r'C:\Data\Vicar\nove_skluzavky_save\predicted2'

segmenter_name=r"C:\Data\Vicar\nove_skluzavky_save\best_models\dt_0.7140693745072867.p"



segmenter = pickle.load( open( segmenter_name, "rb" ) )


data_path=r'C:\Data\Vicar\nove_skluzavky_save\nuc_for_labeling\sample3'

names = glob(data_path + '/*_QPI.tiff')
            
            
            
for name_num,name in enumerate(names):
    
    print(str(name_num)+'/'+str(len(names)))
    
    
    img=imread(name)
    img=img.astype(np.float32)
    
    lam = 0.65
    alpha = 0.18
    img = img * lam / (2 * np.pi * alpha) # o mass
    
    img0=img.copy()

    

    
    shape=np.shape(img)
    img=torch.from_numpy(img.reshape((1,1,shape[0],shape[1])).astype(np.float32))
    

    
    imgs=segmenter.predict_imgs(img)
    res=segmenter.get_segmentation(imgs)
    
    img=img.detach().cpu().numpy()[0,0,:,:]
    
    
    tmp = save_folder + '/' + os.path.split(name)[1]
    
    
    save_name = tmp.replace('_QPI.tiff','') + '_predicted_mask.tif'
    imsave(save_name, (res>0).astype(np.uint8)*255)
    
    
    save_name = tmp.replace('_QPI.tiff','') + '_predicted_dt.tif'
    imsave(save_name, imgs[0])
    
    save_name = tmp.replace('_QPI.tiff','') + '_predicted_fg.tif'
    imsave(save_name, imgs[1].astype(np.float32))
    

      
    
    
    
