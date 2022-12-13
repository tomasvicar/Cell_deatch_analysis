import os
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from shutil import copyfile
import segmentation_models_pytorch as smp
from shutil import rmtree
import sys
from glob import glob
from skimage.io import imread
from skimage.io import imsave
from scipy.signal import convolve2d 


from unet import Unet
from config import Config
from utils.mat2gray import mat2gray



config = Config()
device = torch.device(config.device)

data_dir = r'D:\nove_skluzavky_save\nuc_for_labeling'
save_dir = r'D:\nove_skluzavky_save\nuc_prediction'

model = torch.load(r"C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\best_model\_839_0.00001_train_0.00025_valid_0.00019.pt")

model = model.to(device)

model.eval()


filenames_dapi = glob(data_dir + '/**/*_DAPI.tiff')


for i, filename_dapi in enumerate(filenames_dapi):
    
    print(str(i) + '/' + str(len(filenames_dapi)))
    print(filename_dapi)
    

    filename_qpi = filename_dapi.replace('_DAPI.tiff','_QPI.tiff')
    
    filename_save = filename_dapi.replace('_DAPI.tiff','') + '_prediction.tiff'
    filename_save = filename_save.replace(data_dir,save_dir)
    

    img_dapi = imread(filename_dapi)
    img_qpi = imread(filename_qpi)
    
    img_dapi = img_dapi.astype(np.float64)
    img_dapi = mat2gray(img_dapi, config.norm_dapi)
    
    
    img_qpi = img_qpi.astype(np.float64)
    img_qpi = mat2gray(img_qpi, config.norm_qpi)
    
    img = np.stack((img_dapi, img_qpi), axis=0)
    img = np.expand_dims(img, axis=0)
        
    img = torch.from_numpy(img.astype(np.float32))
    
    
    patch_size = model.config.patch_size  
    border = 75
    
    
    weigth_window=2*np.ones((patch_size,patch_size))
    weigth_window=convolve2d(weigth_window,np.ones((border,border))/np.sum(np.ones((border,border))),'same')
    weigth_window=weigth_window-1
    weigth_window[weigth_window<0.01]=0.01
    
    
    
    img_size = list(img.shape)[2:4]
    
    
    sum_img=np.zeros(img_size)
    count_img=np.zeros(img_size)
    
    
    corners=[]
    cx=0
    while cx<img_size[0]-patch_size: 
        cy=0
        while cy<img_size[1]-patch_size:
            
            corners.append([cx,cy])
            
            cy=cy+patch_size-border
        cx=cx+patch_size-border
       
    cx=0
    while cx<img_size[0]-patch_size:
        corners.append([cx,img_size[1]-patch_size])
        cx=cx+patch_size-border
        
    cy=0
    while cy<img_size[1]-patch_size:
        corners.append([img_size[0]-patch_size,cy])
        cy=cy+patch_size-border   
        
    corners.append([img_size[0]-patch_size,img_size[1]-patch_size])
    
    for corner in corners:
        
        subimg = img[:,:,corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size]
    
        subimg = subimg.to(device)
        with torch.no_grad():
            res=model(subimg)
        
        res = res.detach().cpu().numpy()
        
        
        sum_img[corner[0]:(corner[0]+patch_size),corner[1]:(corner[1]+patch_size)]=sum_img[corner[0]:(corner[0]+patch_size),corner[1]:(corner[1]+patch_size)]+res*weigth_window

        count_img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size]=count_img[corner[0]:corner[0]+patch_size,corner[1]:corner[1]+patch_size]+weigth_window
    
    final=sum_img/count_img
    
    
    
    if not os.path.exists(os.path.split(filename_save)[0]):
        os.makedirs(os.path.split(filename_save)[0])
    
    imsave(filename_save,final)
    
    # new = imread(filename_save)
    
    plt.imshow(final)
    plt.show()
    
    
    # fdsfdf
    
    
    
    





