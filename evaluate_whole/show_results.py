import torch
import os
from skimage.io import imread
from tifffile import TiffWriter
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json
from scipy.interpolate import interp1d


def fillmissing(signal_with_nans):
    
    y = signal_with_nans.copy()
    
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    
    return y


missing_pairs = dict()
missing_pairs['area'] = 'Compensated phase'
missing_pairs['mass'] = 'Compensated phase'
missing_pairs['density'] = 'Compensated phase'
missing_pairs['CDS'] = 'Compensated phase'
missing_pairs['mean_TRITC'] = 'Clipped:TRITC'
missing_pairs['mean_FITC'] = 'Clipped:FITC'
missing_pairs['mean_DAPI'] = 'Clipped:DAPI'
missing_pairs['mean_detection_DAPI'] = 'Clipped:DAPI'



data_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\exported_tiff'
results_path = r'Z:\999992-nanobiomed\Holograf\Tomas_\Lytic Cell Death Distinction\evaluate_whole'


filenames_qpi = glob(data_path + '/**/*Compensated phase.tiff')
filenames_dapi = [x.replace('Compensated phase.tiff','DAPI.tiff') for x in filenames_qpi]


for tiff_num, filename_qpi, in enumerate(filenames_qpi):
    
    print(str(tiff_num) + ' / ' + str(len(filenames_qpi)))
    print(filename_qpi)
    
    
    filename_results = filename_qpi.replace('Compensated phase.tiff','').replace(data_path,results_path) + 'results.json'
        
   
    filename_missing =  filename_qpi.replace('Compensated phase.tiff','missing.json')
        
    with open(filename_results) as json_file:
        results = json.load(json_file)
            
    with open(filename_missing) as json_file:
        missing =json.load(json_file)
        
        
    
    
    for key in results:
        signal = np.array(results[key])
        nans = np.array(missing[missing_pairs[key]])

        if key != 'CDS':
            signal[nans] = np.nan
        else:
            signal[nans] = np.nan
            signal[nans - 1] = np.nan
            signal = np.append(signal,np.nan)
        signal[0:2] = np.nan
            
        signal = fillmissing(signal)
        results[key] = signal
        
        
    
    
    
    
    
    for key in results:
        signal = np.array(results[key])
        plt.plot(signal)
        plt.title(key)
        plt.show()
        
    fgsdgf



    