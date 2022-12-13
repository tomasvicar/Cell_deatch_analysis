from torch.utils import data
import numpy as np
import torch 
import os
from skimage.io import imread
from glob import glob

from scipy.io import loadmat

from scipy.ndimage import gaussian_filter
from scipy.ndimage import laplace

import matplotlib.pyplot as plt
# import cv2
from utils.mat2gray import mat2gray



def remove_too_close(detections, mindist):
    
    remove = []
    for row in range(detections.shape[0]):
        if row not in remove:

            dist = np.sqrt((detections[:, 0] - detections[row, 0]) ** 2  + (detections[:, 1] - detections[row, 1]) ** 2 )
            dist[row] = 999
            remove.extend((np.where(dist < mindist)[0]).tolist())
        
    
    detections = np.delete(detections, remove, axis=0)
    return detections



class Augmenter():
    
    def __init__(self, cols, rows):
    
        def rand():
            return torch.rand(1).numpy()[0]
        
        self.cols = cols
        self.rows = rows
        sr = 0.2
        gr = 0.05
        tr = 0
        dr = 30
        rr = 180
        sx = 1 + sr * rand()
        if rand() > 0.5:
            sx = 1 / sx
        sy = 1 + sr * rand()
        if rand() > 0.5:
            sy = 1 / sy
        gx = (0 - gr) + gr * 2 * rand()
        gy = (0 - gr) + gr * 2 * rand()
        tx = (0 - tr) + tr * 2 * rand()
        ty = (0 - tr) + tr * 2 * rand()
        dx = (0 - dr) + dr * 2 * rand()
        dy = (0 - dr) + dr * 2 * rand()
        t  = (0 - rr) + rr * 2 *rand()
        
        # M = np.array([[sx, gx, dx], [gy, sy, dy],[tx, ty, 1]])
        # R = cv2.getRotationMatrix2D((cols / 2, rows / 2), t, 1)
        # R = np.concatenate((R, np.array([[0, 0, 1]])), axis=0)
        # self.matrix = np.matmul(R, M)
        self.r = [torch.randint(2, (1, 1)).view(-1).numpy(),torch.randint(2, (1, 1)).view(-1).numpy(),torch.randint(4, (1, 1)).view(-1).numpy()]
        
        
    def augment(self, img, is_gt, nn_interp):
        
        def rand():
            return torch.rand(1).numpy()[0]
        
        # if nn_interp:
        #     img = cv2.warpPerspective(img, self.matrix, (self.cols, self.rows), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # else:
        #     img = cv2.warpPerspective(img, self.matrix, (self.cols, self.rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        
        if self.r[0]:
            img = np.fliplr(img)
        if self.r[1]:
            img = np.flipud(img)
        img = np.rot90(img, k = self.r[2])    
        
        if not is_gt:
            multipy = 0.2 
            multipy = 1 + rand() * multipy
            if rand() > 0.5:
                img = img * multipy
            else:
                img = img / multipy
               
            add = 0.2    
            add = (1 - 2 * rand()) * add
            img = img + add
        
        if not is_gt:
            bs_r = (-0.5, 0.5)
            r = 1 - 2 * rand()
            if r <= 0:
                par = bs_r[0] * r
                img = img - par * laplace(img)
            if r > 0:
                par = bs_r[1] *r
                img = gaussian_filter(img, par)
        
        return img







class Dataset(data.Dataset):


    def __init__(self, names, augment, crop, config, crop_same=False):
       
        self.names = names
        self.augment = augment
        self.crop = crop
        self.config = config
        self.crop_same = crop_same

    def __len__(self):
        return len(self.names['lbls'])


    def __getitem__(self, idx):

        name_lbl = self.names['lbls'][idx]
        name_img_dapi = self.names['imgs_dapi'][idx]
        name_img_qpi = self.names['imgs_qpi'][idx]
        
            
        img_dapi = imread(name_img_dapi)
        img_dapi = img_dapi.astype(np.float64)
        img_dapi = mat2gray(img_dapi, self.config.norm_dapi)
        
        
        img_qpi = imread(name_img_qpi)
        img_qpi = img_qpi.astype(np.float64)
        img_qpi = mat2gray(img_qpi, self.config.norm_qpi)
        
        
        detections_tmp = loadmat(name_lbl)['detections']
        # detections_tmp = np.concatenate((detections_tmp, detections_tmp),axis=0)
        detections = remove_too_close(detections_tmp, self.config.too_close)
        detections = np.round(detections).astype(int)
        lbl = np.zeros_like(img_dapi)
        lbl[detections[:, 1], detections[:, 0]] = 1 
        
        
        if self.augment:
            augmenter = Augmenter(img_dapi.shape[0], img_dapi.shape[1])
            img_dapi = augmenter.augment(img_dapi, is_gt=False, nn_interp = False)
            img_qpi = augmenter.augment(img_qpi, is_gt=False, nn_interp = False)
            lbl = augmenter.augment(lbl, is_gt=True, nn_interp = True)
        
        lbl = gaussian_filter(lbl, self.config.sigma) / self.config.div 
        
        in_size = img_qpi.shape
        out_size = [self.config.patch_size, self.config.patch_size]
        
        
        if self.crop:
            r1 = torch.randint((in_size[0] - out_size[0]), (1, 1)).view(-1).numpy()[0]
            r2 = torch.randint((in_size[1] - out_size[1]), (1, 1)).view(-1).numpy()[0]
            r = [r1 ,r2]
            
            if self.crop_same:
                r = [100,100]
            
            
            img_dapi = img_dapi[r[0] : (r[0] + out_size[0]) ,r[1] : (r[1] + out_size[1])]   
            img_qpi = img_qpi[r[0] : (r[0] + out_size[0]) ,r[1] : (r[1] + out_size[1])]   
            lbl = lbl[r[0] : (r[0] + out_size[0]) ,r[1] : (r[1] + out_size[1])]   

        img = np.stack((img_dapi, img_qpi), axis=0)
        # img = np.expand_dims(img_dapi, axis=0)
            
        img = torch.from_numpy(img.astype(np.float32))
        lbl = torch.from_numpy(np.expand_dims(lbl, axis=0).astype(np.float32))
        
        return img, lbl




if __name__ == "__main__":
    from config import Config
    import sys
    from load_and_split_data import load_and_split_data
    config = Config()
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        save_dir = sys.argv[2]
        
    else:
        data_dir = r'D:\nove_skluzavky_save\nuc_for_labeling'
        save_dir = '../../tmp'
        
        config.train_batch_size = 2
        config.train_num_workers = 0
        config.valid_batch_size = 2
        config.valid_num_workers = 0
    
    
    config.data_path = data_dir
    config.save_dir = save_dir
    config.best_model_dir = config.save_dir  + '/best_model'
    
    names = load_and_split_data(config)    
    
    
    generator = Dataset(names['names_train'], augment=True, crop=True, config=config)
    generator = data.DataLoader(generator, batch_size=config.train_batch_size,num_workers=config.train_num_workers, shuffle=True, drop_last=True)
    
    # generator = Dataset(names['names_valid'], augment=False, crop=True, config=config)
    # generator = data.DataLoader(generator, batch_size=config.train_batch_size,num_workers=config.train_num_workers, shuffle=True, drop_last=True)
    
    # generator = Dataset(names['names_test'], augment=False, crop=False, config=config)
    # generator = data.DataLoader(generator, batch_size=config.train_batch_size,num_workers=config.train_num_workers, shuffle=True, drop_last=True)
    

    for img, lbl in generator:
        
        plt.imshow(img[0,0,:,:], vmin=0, vmax=1)
        plt.show()
        plt.imshow(img[0,1,:,:], vmin=0, vmax=1)
        plt.show()
        plt.imshow(lbl[0,0,:,:], vmin=0, vmax=1)
        plt.show()
        
        break


