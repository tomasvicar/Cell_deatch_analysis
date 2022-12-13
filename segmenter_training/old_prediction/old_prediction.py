from glob import glob
import torch
import segmentation_models_pytorch as smp 
from skimage.io import imread
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
import os

# pip install segmentation-models-pytorch==0.1.3
# 
#  + prepsano 
# from torch._six import container_abcs
# 
# na 
# try:
#     from torch._six import container_abcs
# except:
#     import collections.abc as container_abcs
#  
#  v C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\env\lib\site-packages\timm\models\layers\helpers.py




device = torch.device('cuda:0')


data_path = r'C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\nuc_for_labeling\sample3'
save_path = r'C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\cell_prediction_1\sample3'



model = torch.load(r"C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky\segmenter_training\old_models\dt_54_0.00100_gpu_6.09328_train_0.00591_valid_0.00315.pt")
model.eval()
model=model.to(device)



model_semantic = torch.load(r"C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky\segmenter_training\old_models\semantic_71_0.00001_gpu_6.09328_train_0.06072_valid_0.04296.pt")
model_semantic.eval()
model_semantic=model_semantic.to(device)


file_names = glob(data_path + '/*QPI.tiff')

for file_num, file_name in enumerate(file_names):
    
    
    img = imread(file_name)
    
    img = torch.from_numpy(img.astype(np.float32))
    
    
    lam = 0.65
    alpha = 0.18
    img = img * lam / (2 * np.pi * alpha) # o mass
    
    img = torch.unsqueeze(torch.unsqueeze(img, 0), 0)

    img = img.to(device)
    

    with torch.no_grad():
        img = F.pad(img,(0,40,0,40),'reflect')
        res = model(img)
        res = res[:,:,:-40,:-40]
        dt = res.detach().cpu().numpy()[0,0,:,:]
        
        res = model_semantic(img)
        res = res[:,:,:-40,:-40]
        seg = res.detach().cpu().numpy()[0,0,:,:]>0
        
        img = img.detach().cpu().numpy()[0,0,:-40,:-40]
        
        
    
    plt.imshow(img)
    plt.show()
    plt.imshow(dt)
    plt.show()
    plt.imshow(seg)
    plt.show()
    
    
    filename_save = file_name.replace('_QPI.tiff','') + '_prediction_dt.tiff'
    filename_save = filename_save.replace(data_path, save_path)
    
    if not os.path.exists(os.path.split(filename_save)[0]):
        os.makedirs(os.path.split(filename_save)[0])
    
    imsave(filename_save, dt)

    

    filename_save = file_name.replace('_QPI.tiff','') + '_prediction_fg.tiff'
    filename_save = filename_save.replace(data_path, save_path)
    
    if not os.path.exists(os.path.split(filename_save)[0]):
        os.makedirs(os.path.split(filename_save)[0])
    
    imsave(filename_save, (seg * 255).astype(np.uint8))
    
     















