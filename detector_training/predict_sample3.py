from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread
from scipy.signal import convolve2d 
import torch
from multiprocessing import Pool
import json
import time

from utils.mat2gray import mat2gray
from detect import detect





with open('detection_params.json', 'r') as f:
    detection_params = json.load(f)['params']



filenames = glob(r'C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\nuc_prediction3\sample3' + '/*_prediction.tiff')



for tiff_num, filename in enumerate(filenames):
    
    print(str(tiff_num) + ' / ' + str(len(filenames)))
    print(filename)
    

    img= imread(filename)
    
    
    detection = detect(img, T=detection_params['T'], h=detection_params['h'], d=detection_params['d'])

    
    save_name = filename.replace('nuc_prediction3','nuc_detection').replace('prediction.tiff', '') + 'detections.json'
    
    folder = os.path.split(save_name)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        
    detection = detection.tolist() 
    
    detection_dict = dict()
    detection_dict['detection'] = detection
    
    with open(save_name, 'w') as f:
        f.write(json.dumps(detection_dict))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

