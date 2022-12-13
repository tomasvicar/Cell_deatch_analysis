import torch
import os
from skimage.io import imread
from tifffile import TiffWriter
import numpy as np
import matplotlib.pyplot as plt
from utils.visboundaries import visboundaries
from utils.colorize_notouchingsamecolor import colorize_notouchingsamecolor
import pickle
from utils.get_jacards_cell import get_jacards_cell,get_jacards_cell_with_fp
from glob import glob
import torch.nn.functional as F



segmenter_name = r"C:\Users\tomas\Desktop\nove_skluzavky\best_models_training1\dt_0.7140693745072867.p"
dt_model_name = r"C:\Users\tomas\Desktop\nove_skluzavky\best_models_training1\dt_61_0.00010_gpu_0.16608_train_0.00564_valid_0.00279.pt"
fg_model_name = r"C:\Users\tomas\Desktop\nove_skluzavky\best_models_training1\semantic_71_0.00001_gpu_0.16608_train_0.05204_valid_0.03319.pt"


segmenter = pickle.load( open( segmenter_name, "rb" ) )


device = torch.device('cuda:0')
    
model_dt = torch.load(dt_model_name)
model_dt.eval()
model_dt = model_dt.to(device)

model_fg = torch.load(fg_model_name)
model_fg.eval()
model_fg = model_fg.to(device)

data_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\exported_tiff'
save_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\segmentation1'



filenames_qpi = glob(data_path + '/**/*Compensated phase.tiff')
filenames_dapi = [x.replace('Compensated phase.tiff','DAPI.tiff') for x in filenames_qpi]


for tiff_num, (filename_qpi, filename_dapi) in enumerate(zip(filenames_qpi, filenames_dapi)):
    
    print(str(tiff_num) + ' / ' + str(len(filenames_qpi)))
    print(filename_qpi)
    
    
    img_all = imread(filename_qpi,key = slice(None)).astype(np.float32)
    
    dts = np.zeros_like(img_all)
    fgs = np.zeros_like(img_all)
    
    lam = 0.65
    alpha = 0.18
    img_all = img_all * lam / (2 * np.pi * alpha) # o mass
    
    
    shape=np.shape(img_all)
    
    
    ts = img_all.shape[0]
    z_step = 1

    
    for c_z in range(0, ts, z_step):
        if c_z + z_step <= ts:
            inds_z =  slice(c_z, c_z + z_step)
        else:
            inds_z =  slice(c_z, ts)
    

        img = img_all[inds_z , :, :]
        
        img=torch.from_numpy(img.reshape((img.shape[0],1,shape[1],shape[2])).astype(np.float32))
    
        
        img = img.to(device)
        
        img = F.pad(img,(0,40,0,40),'reflect')
        res = model_dt(img)
        res = res[:,:,:-40,:-40]
        dt = res.detach().cpu().numpy()[:,0,:,:]
        
        res = model_fg(img)
        res = res[:,:,:-40,:-40]
        seg = res.detach().cpu().numpy()[:,0,:,:]>0

        dts[inds_z , :, :] = dt
        fgs[inds_z , :, :] = seg


    res_all = np.zeros_like(img_all,dtype=bool)
    for frame_num, (dt, seg) in enumerate(zip(dts, fgs)):
        res_all[frame_num,:,:] = segmenter.get_segmentation([dt,seg])>0
        



    save_name = save_path + filename_qpi.replace(data_path,'').replace('Compensated phase.tiff', '') + 'segmentation.tiff'
    save_name_fg = save_path + filename_qpi.replace(data_path,'').replace('Compensated phase.tiff', '') + 'fg.tiff'
    
    folder = os.path.split(save_name)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        
    fgs = fgs > 0;
    with TiffWriter(save_name_fg,bigtiff=True) as tif:
        for fg in fgs:
            tif.write(fg)
    
    
    with TiffWriter(save_name,bigtiff=True) as tif:

        for res in res_all:
    
            tif.write(res)
    

