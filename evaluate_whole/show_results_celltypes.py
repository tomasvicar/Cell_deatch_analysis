import torch
import os
from skimage.io import imread
from tifffile import TiffWriter
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

def fillmissing(signal_with_nans):
    
    y = signal_with_nans.copy()
    
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    
    return y




data_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\exported_tiff'
results_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\evaluate_whole'


filenames_qpi = glob(data_path + '/**/*Compensated phase.tiff')
filenames_dapi = [x.replace('Compensated phase.tiff','DAPI.tiff') for x in filenames_qpi]


results_all = dict()
for tiff_num, filename_qpi, in enumerate(filenames_qpi):
    
    print(str(tiff_num) + ' / ' + str(len(filenames_qpi)))
    print(filename_qpi)
    
    
    filename_results = filename_qpi.replace('Compensated phase.tiff','').replace(data_path,results_path) + 'results.json'
        

        
    with open(filename_results) as json_file:
        results = json.load(json_file)
        
    results_all[os.path.split(filename_qpi)[-1].replace('_Compensated phase.tiff','')] = results
    
    features_keys = list(results.keys())
    
            

all_keys = list(results_all.keys())
all_keys_main = list(set([x[:-4] for x in all_keys]))
    
    
for cell_type in ['PC3', 'FaDu']:
    

    result_avg = dict()
    for key_main in all_keys_main:
        
        if cell_type not in key_main:
            continue
        
        result_avg[key_main] = dict()
        for features_key in features_keys:
            result_avg[key_main][features_key] = []
        
        for key in all_keys:
            
            if key_main not in key:
                continue
            
            for features_key in features_keys:
                tmp = np.array(results_all[key][features_key])
                tmp = medfilt(tmp,5)
                tmp = gaussian_filter1d(tmp,1)
                result_avg[key_main][features_key].append(tmp)
                
                
    
    for features_key in features_keys:
        
        plt.figure(figsize=[20,20])
        used_key_main = []
        for key_main in all_keys_main:

            if cell_type not in key_main:
                continue
            
            used_key_main.append(key_main)
            
            tmp = result_avg[key_main][features_key]
            tmp = np.stack(tmp,axis=0)
            tmp = np.median(tmp,axis=0)
            
            plt.plot(tmp, label=key_main)
            
        

        plt.title(cell_type + '   '  + features_key)
        plt.legend()
        plt.savefig('results' + '/' + cell_type + '_' + features_key + '_plot.png',  bbox_inches='tight', pad_inches=0)
        plt.show()
    
        
        
    
            
            
            
            
    
        
    



    