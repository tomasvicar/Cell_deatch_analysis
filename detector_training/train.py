import os
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from shutil import copyfile
import segmentation_models_pytorch as smp
from shutil import rmtree
import sys

from dataset import Dataset
from unet import Unet
from utils.log import Log
from utils.training_fcns import l1_loss,l2_loss,dice_loss_logit
from config import Config
from load_and_split_data import load_and_split_data



def train(config, names):

    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.isdir(config.best_models_dir):
        os.mkdir(config.best_models_dir)
    
    device = torch.device(config.device)

    train_generator = Dataset(names['names_train'], augment=True, crop=True, config=config)
    train_generator = data.DataLoader(train_generator, batch_size=config.train_batch_size,num_workers=config.train_num_workers, shuffle=True, drop_last=True)

    valid_generator = Dataset(names['names_valid'], augment=False, crop=True, config=config)
    valid_generator = data.DataLoader(valid_generator, batch_size=config.valid_batch_size, num_workers=config.valid_num_workers, shuffle=True ,drop_last=True)

    
    model = smp.Unet(
        encoder_name="efficientnet-b2",
        encoder_weights='imagenet',
        in_channels=2,
        classes=1,
    )
    
    # model = Unet(filters=[16, 32, 64, 128, 256], in_size=2,out_size=1)
    
    model.config = config
    model.names = names
        
        
    model = model.to(device)
    
    model.log = Log()

    optimizer = torch.optim.AdamW(model.parameters(),lr =config.init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_changes_list, gamma=config.gamma, last_epoch=-1)

    model_names=[]
    for epoch in range(config.max_epochs):
        
        if epoch == 1:
            model.log = Log()
        
        model.train()
        for img, lbl in train_generator:
            
            img = img.to(device)
            
            res = model(img)
            
            loss = l2_loss(res, lbl)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.log.save_tmp_log(loss,'train')
                
            
        model.eval()
        with torch.no_grad():
            for img, lbl in valid_generator:
                
                img = img.to(device)
                
                res = model(img)
                
                loss = l2_loss(res, lbl)
                
                model.log.save_tmp_log(loss,'valid')
            
        
        model.log.save_log_data_and_clear_tmp()
        
        model.log.plot_training()
        
            
        res = res.detach().cpu().numpy()
        lbl = lbl.detach().cpu().numpy()
        img = img.detach().cpu().numpy()
        for k in range(res.shape[0]):
            plt.imshow(np.concatenate((img[k,0,:,:], res[k,0,:,:], lbl[k,0,:,:]),axis=1),vmin=0,vmax=1)
            plt.show()
            plt.close()
    
    
        xstr = lambda x:"{:.5f}".format(x)
        lr=optimizer.param_groups[0]['lr']
        info= '_' + str(epoch) + '_' + xstr(lr) + '_train_'  + xstr(model.log.train_loss_log[-1]) + '_valid_' + xstr(model.log.valid_loss_log[-1]) 
        print(info)
        
        model_name=config.save_dir+ os.sep + info  + '.pt'
        
        model_names.append(model_name)
        
        torch.save(model,model_name)
        
        model.log.plot_training(model_name.replace('.pt','loss.png'))
        
        scheduler.step()

    
    best_model_ind = np.argmin(model.log.valid_loss_log)
    best_model_name = model_names[best_model_ind]   
    best_model_name_new = best_model_name.replace(config.save_dir, config.best_models_dir)
    
    copyfile(best_model_name,best_model_name_new)
    


if __name__ == "__main__":
    
    config = Config()
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        save_dir = sys.argv[2]
        
    else:
        data_dir = r'C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\nuc_for_labeling'
        save_dir = r'D:\nove_skluzavky_save'
        
        config.train_batch_size = 4
        config.train_num_workers = 2
        config.valid_batch_size = 2
        config.valid_num_workers = 1
    
    
    config.data_path = data_dir
    config.save_dir = save_dir
    config.best_models_dir = save_dir + '/../best_model'
    
    names = load_and_split_data(config)

    train(config, names)
    
    
    

